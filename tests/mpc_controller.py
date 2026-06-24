#!/usr/bin/env python3
"""
mpc_controller.py — Linear MPC for lateral lane-change maneuvers
================================================================
Kinematic bicycle model, linearised around (θ≈0, δ≈0, v≈v₀),
discretised at Ts.  QP solved via scipy.optimize.minimize (SLSQP)
with a pure-numpy projected-gradient fallback.

Vehicle parameters
------------------
    Wheelbase   L  = 0.15 m
    Width       W  = 0.18 m   (used for lane-bound constraints only)
    Max steer   δ_max = 25°

Model  (continuous, linearised)
------
    State   z = [y, θ]ᵀ          lateral position (m), heading (rad)
    Input   u = [δ, v]ᵀ          steering angle (rad), speed (m/s)

    ẏ  = v₀ · θ
    θ̇  = v₀ · δ / L

    A_c = [[0, v₀],              B_c = [[0      ],
           [0,  0]]                     [v₀ / L ]]

Discretised  (exact ZOH)
-------------------------
    A_d = [[1, v₀·Ts],           B_d = [[v₀²·Ts² / (2L)],
           [0,     1]]                   [v₀·Ts  / L    ]]

QP  (condensed form)
--------------------
    Decision vector   x = [δ₀ … δ_{M-1},  v₀ … v_{M-1},  ε]
                           ───────────── ──────────────  ─
                            M steers       M speeds      slack

    min  ½ xᵀ H x + fᵀ x
    s.t. |δ_k| ≤ δ_max            (hard input)
         v_min ≤ v_k ≤ v_max      (hard input)
         y_min − ε ≤ ŷ_k ≤ y_max + ε   (soft output)
         ε ≥ 0
"""

import numpy as np
import math
import time

try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[mpc] scipy not found — using numpy projected-gradient fallback")


# ═══════════════════════════════════════════════════════════════════════════════
class LaneChangeMPC:
    """
    Linear MPC for lane-change trajectory tracking.

    Typical call pattern (inside control_loop):
        mpc = LaneChangeMPC()
        ...
        delta_rad, v_ms = mpc.solve(z0, s_now, D, LANE_WIDTH, +1, v0, v_ref, u_prev)
        steer_cmd = delta_rad / mpc.delta_max     # → [-1, 1]
        throttle  = nominal_throttle * (v_ms / v_nominal)
    """

    def __init__(
        self,
        Ts: float = 0.05,           # sample time  (20 Hz)
        P: int = 10,                 # prediction horizon  (25 × 0.05 = 1.25 s)
        M: int = 5,                  # control horizon  (16 % of P)
        L: float = 0.15,            # wheelbase  (m)
        delta_max_deg: float = 25.0, # max steering angle  (°)
        # ── cost weights ──
        q_y: float = 100.0,         # lateral tracking
        r_delta: float = 1.0,       # steering rate  (Δδ penalty)
        r_v: float = 10.0,          # speed deviation penalty
        rho: float = 1000.0,        # soft-constraint penalty
        # ── limits ──
        v_min: float = 0.05,        # min speed  (m/s)
        v_max: float = 0.30,        # max speed  (m/s)
        y_margin: float = 0.05,     # soft output margin  (m)
        max_iter: int = 15,        # max solver iterations
    ):
        self.Ts = Ts
        self.P = P
        self.M = M
        self.L = L
        self.delta_max = math.radians(delta_max_deg)

        self.q_y = q_y
        self.r_delta = r_delta
        self.r_v = r_v
        self.rho = rho

        self.v_min = v_min
        self.v_max = v_max
        self.y_margin = y_margin
        self.max_iter = max_iter

        # Dimensions
        self.nx = 2        # [y, θ]
        self.n_delta = M
        self.n_v = M
        self.n_slack = 1
        self.n_vars = M + M + 1   # total decision variables

        # ── Pre-build differencing matrix T for Δδ ────────────────────────
        #    Δδ₀ = δ₀ − δ_prev   (δ_prev enters via t₀ vector)
        #    Δδₖ = δₖ − δₖ₋₁     for k ≥ 1
        self.T_diff = np.zeros((M, M))
        for i in range(M):
            self.T_diff[i, i] = 1.0
            if i > 0:
                self.T_diff[i, i - 1] = -1.0

        # ── Pre-build output-selection matrix C_y ─────────────────────────
        #    Picks y (index 0) from each [y, θ] block in the stacked prediction
        self.C_y = np.zeros((P, self.nx * P))
        for k in range(P):
            self.C_y[k, k * self.nx] = 1.0

        # ── Cached prediction matrices (rebuilt when v₀ changes) ──────────
        self._cached_v0 = None
        self._Ad = None
        self._Bd = None
        self._Psi = None
        self._Theta = None
        self._Psi_y = None     # C_y @ Psi   →  (P × nx)
        self._Theta_y = None   # C_y @ Theta  →  (P × M)

        # ── Diagnostics ──────────────────────────────────────────────────
        self.last_solve_ms = 0.0
        self.last_cost = 0.0
        self.last_y_pred = None   # predicted y-trajectory (P,)

    # ──────────────────────────────────────────────────────────────────────
    #  Model matrices
    # ──────────────────────────────────────────────────────────────────────
    def _build_dynamics(self, v0: float):
        """Discrete A_d, B_d for the linearised bicycle model at speed v₀."""
        Ts, L = self.Ts, self.L
        Ad = np.array([[1.0, v0 * Ts],
                       [0.0, 1.0]])
        Bd = np.array([[v0**2 * Ts**2 / (2.0 * L)],
                       [v0 * Ts / L]])
        return Ad, Bd

    def _build_prediction(self, Ad, Bd):
        """
        Condensed prediction matrices.

        Z = Ψ·z₀ + Θ·U_δ

        Z  = [z₁ … z_P]ᵀ           (nx·P × 1)
        U_δ = [δ₀ … δ_{M-1}]ᵀ      (M × 1)

        For k ≥ M the input is held at δ_{M-1}  (control-horizon constraint).
        """
        P, M, nx = self.P, self.M, self.nx

        Psi = np.zeros((nx * P, nx))
        Theta = np.zeros((nx * P, M))

        # Pre-compute Ad powers: Ad_pow[i] = Ad^i
        Ad_pow = [np.eye(nx)]
        for _ in range(P):
            Ad_pow.append(Ad_pow[-1] @ Ad)

        for k in range(P):
            # Free response
            Psi[k * nx:(k + 1) * nx, :] = Ad_pow[k + 1]

            # Forced response
            for j in range(min(k + 1, M)):
                Theta[k * nx:(k + 1) * nx, j:j + 1] += Ad_pow[k - j] @ Bd

            # Hold past control horizon: u_j = u_{M-1} for j = M … k
            for j in range(M, k + 1):
                Theta[k * nx:(k + 1) * nx, M - 1:M] += Ad_pow[k - j] @ Bd

        return Psi, Theta

    def _update_matrices(self, v0: float):
        """Rebuild prediction matrices if v₀ changed by > 5 mm/s."""
        if self._cached_v0 is not None and abs(v0 - self._cached_v0) < 0.005:
            return
        self._cached_v0 = v0
        self._Ad, self._Bd = self._build_dynamics(v0)
        self._Psi, self._Theta = self._build_prediction(self._Ad, self._Bd)
        self._Psi_y = self.C_y @ self._Psi       # (P × nx)
        self._Theta_y = self.C_y @ self._Theta   # (P × M)

    # ──────────────────────────────────────────────────────────────────────
    #  Reference trajectory
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def quintic_ref(s: float, D: float, lane_width: float, direction: float) -> float:
        """
        Quintic smooth-step + asymmetric kick reference.

        Parameters
        ----------
        s         : distance along maneuver  (m)
        D         : total maneuver distance  (m)
        lane_width: absolute lateral shift   (m)
        direction : +1.0  (overtake, shift left)
                    −1.0  (recovery, shift right)
        """
        if s <= 0.0:
            return 0.0
        if s >= D:
            return direction * lane_width
        r = s / D
        poly = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
        kick = (r ** 0.5) * (1.0 - r)
        poly += 0.4 * kick
        return direction * lane_width * poly

    # ──────────────────────────────────────────────────────────────────────
    #  QP construction & solve
    # ──────────────────────────────────────────────────────────────────────
    def solve(
        self,
        z0: np.ndarray,       # [y, θ]  current state
        s_now: float,         # encoder distance from maneuver start  (m)
        D: float,             # total maneuver distance  (m)
        lane_width: float,    # absolute lateral shift   (m)
        direction: float,     # +1 overtake, −1 recovery
        v0: float,            # current measured speed   (m/s)
        v_ref: float,         # desired speed            (m/s)
        u_prev_delta: float,  # previous steering angle  (rad)
    ):
        """
        Solve the MPC QP.  Returns (δ_opt_rad, v_opt_ms).

        Only the *first* element of each control sequence is returned
        (receding-horizon principle).
        """
        t0 = time.time()

        v0 = max(v0, 0.01)
        self._update_matrices(v0)

        P, M = self.P, self.M
        n = self.n_vars           # 2M + 1

        # ── Reference trajectory over prediction horizon ──────────────────
        Y_ref = np.empty(P)
        for k in range(P):
            s_future = s_now + v0 * (k + 1) * self.Ts
            Y_ref[k] = self.quintic_ref(s_future, D, lane_width, direction)

        # ── Free response (y when δ = 0 throughout) ──────────────────────
        Y_free = self._Psi_y @ z0   # (P,)

        Th_y = self._Theta_y        # (P × M)

        # ==================================================================
        #  Build quadratic cost   ½ xᵀ H x + fᵀ x
        # ==================================================================

        # ── 1. Lateral tracking: ‖Th_y·δ + Y_free − Y_ref‖²_Q ───────────
        Q_diag = self.q_y * np.ones(P)
        # H_dd_lat = Th_yᵀ diag(Q) Th_y
        H_dd_lat = (Th_y.T * Q_diag) @ Th_y          # (M × M)
        residual = Y_free - Y_ref                      # (P,)
        f_d_lat = (Th_y.T * Q_diag) @ residual        # (M,)

        # ── 2. Steering rate: ‖T·δ + t₀‖²_R ─────────────────────────────
        t0_vec = np.zeros(M)
        t0_vec[0] = -u_prev_delta
        R_d_diag = self.r_delta * np.ones(M)
        T = self.T_diff
        H_dd_rate = (T.T * R_d_diag) @ T              # (M × M)
        f_d_rate = (T.T * R_d_diag) @ t0_vec          # (M,)

        # ── Combined δ-block ──────────────────────────────────────────────
        H_dd = H_dd_lat + H_dd_rate                    # (M × M)
        f_d = f_d_lat + f_d_rate                       # (M,)

        # ── 3. Speed tracking: ‖v − v_ref‖²_rv ──────────────────────────
        v_ref_vec = v_ref * np.ones(M)
        H_vv = self.r_v * np.eye(M)                    # (M × M)
        f_v = -self.r_v * v_ref_vec                    # (M,)

        # ── Assemble full H, f ────────────────────────────────────────────
        H = np.zeros((n, n))
        f = np.zeros(n)

        H[:M, :M] = 2.0 * H_dd
        f[:M] = 2.0 * f_d

        H[M:2*M, M:2*M] = 2.0 * H_vv
        f[M:2*M] = 2.0 * f_v

        H[2*M, 2*M] = 2.0 * self.rho      # slack penalty
        # f[2*M] = 0  already

        # ── Bounds (hard input constraints) ───────────────────────────────
        lb = np.empty(n)
        ub = np.empty(n)
        lb[:M] = -self.delta_max            # δ lower
        ub[:M] = self.delta_max             # δ upper
        lb[M:2*M] = self.v_min              # v lower
        ub[M:2*M] = self.v_max              # v upper
        lb[2*M] = 0.0                       # ε ≥ 0
        ub[2*M] = 1e6                       # ε unbounded above

        # ── Soft output constraints  ──────────────────────────────────────
        #    y_min − ε ≤ ŷ_k ≤ y_max + ε
        if direction > 0:   # overtake left  (y goes 0 → +lane_width)
            y_lo = -self.y_margin
            y_hi = lane_width + self.y_margin
        else:               # recovery right  (y goes 0 → −lane_width)
            y_lo = -(lane_width + self.y_margin)
            y_hi = self.y_margin

        # Build inequality matrices for soft-constrained outputs
        #   Th_y[k,:] @ δ + Y_free[k] ≤ y_hi + ε
        #     → [Th_y[k,:], 0…0, -1] @ x ≤ y_hi - Y_free[k]
        #
        #   Th_y[k,:] @ δ + Y_free[k] ≥ y_lo - ε
        #     → [-Th_y[k,:], 0…0, -1] @ x ≤ -y_lo + Y_free[k]
        n_ineq = 2 * P
        A_ineq = np.zeros((n_ineq, n))
        b_ineq = np.zeros(n_ineq)

        for k in range(P):
            # upper bound
            A_ineq[k, :M] = Th_y[k, :]
            A_ineq[k, 2*M] = -1.0          # −ε
            b_ineq[k] = y_hi - Y_free[k]

            # lower bound
            A_ineq[P + k, :M] = -Th_y[k, :]
            A_ineq[P + k, 2*M] = -1.0      # −ε
            b_ineq[P + k] = -y_lo + Y_free[k]

        # ── Solve ─────────────────────────────────────────────────────────
        x0 = np.zeros(n)
        x0[:M] = np.clip(u_prev_delta, -self.delta_max, self.delta_max)
        x0[M:2*M] = np.clip(v_ref, self.v_min, self.v_max)

        if HAS_SCIPY:
            x_opt = self._solve_scipy(H, f, lb, ub, A_ineq, b_ineq, x0, self.max_iter)
        else:
            x_opt = self._solve_pgd(H, f, lb, ub, A_ineq, b_ineq, x0, self.max_iter)

        # ── Extract first commands ────────────────────────────────────────
        delta_opt = float(np.clip(x_opt[0], -self.delta_max, self.delta_max))
        v_opt = float(np.clip(x_opt[M], self.v_min, self.v_max))

        # ── Diagnostics ──────────────────────────────────────────────────
        self.last_solve_ms = (time.time() - t0) * 1000.0
        self.last_cost = float(0.5 * x_opt @ H @ x_opt + f @ x_opt)
        self.last_y_pred = Th_y @ x_opt[:M] + Y_free

        return delta_opt, v_opt

    # ──────────────────────────────────────────────────────────────────────
    #  Solvers
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _solve_scipy(H, f, lb, ub, A_ineq, b_ineq, x0, max_iter):
        """Solve QP via scipy SLSQP."""
        bounds = list(zip(lb, ub))

        # SLSQP uses  g(x) ≥ 0  convention  →  b_ineq − A_ineq @ x ≥ 0
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, A=A_ineq, b=b_ineq: b - A @ x,
            'jac': lambda x, A=A_ineq: -A,
        }]

        result = scipy_minimize(
            fun=lambda x: 0.5 * x @ H @ x + f @ x,
            x0=x0,
            jac=lambda x: H @ x + f,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-3, 'disp': False},
        )
        
        if not result.success:
            print(f"[MPC Warning] Suboptimal solution: {result.message} (iters: {result.nit})")
            
        return result.x

    @staticmethod
    def _solve_pgd(H, f, lb, ub, A_ineq, b_ineq, x0,
                   max_iter: int = 500,
                   penalty_weight: float = 1000.0):
        """
        Projected Gauss-Seidel (Coordinate Descent) fallback solver.
        Extremely robust to ill-conditioned QPs compared to plain gradient descent.
        Handles inequality constraints via an exact penalty method.
        """
        n = len(x0)
        x = x0.copy()
        H_diag = np.diag(H)

        for _ in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                # 1. Base gradient & Hessian for coordinate i
                grad_i = np.dot(H[i, :], x) + f[i]
                hess_i = H_diag[i]

                # 2. Augmented Lagrangian penalty for inequality violations
                #    J_pen = pw * sum( max(0, A_ineq @ x - b_ineq)^2 )
                violation = A_ineq @ x - b_ineq
                active = violation > 0
                if np.any(active):
                    # Derivative of penalty w.r.t x_i
                    grad_i += 2.0 * penalty_weight * np.dot(violation[active], A_ineq[active, i])
                    # Second derivative (approximate Hessian for the penalty)
                    hess_i += 2.0 * penalty_weight * np.sum(A_ineq[active, i]**2)

                # 3. Gauss-Seidel update step
                if hess_i > 1e-8:
                    x[i] -= grad_i / hess_i

                # 4. Project onto box bounds
                x[i] = np.clip(x[i], lb[i], ub[i] if ub[i] < 1e5 else x[i])

            # Early stopping if converged
            if np.max(np.abs(x - x_old)) < 1e-4:
                break

        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MPC Controller — self-test")
    print("=" * 60)

    mpc = LaneChangeMPC()

    # Simulate an overtake from rest
    LANE_WIDTH = 0.112    # 0.28 * 0.4
    D = 0.30              # maneuver distance
    v0 = 0.08             # m/s
    v_ref = 0.08
    z0 = np.array([0.0, 0.0])
    u_prev = 0.0

    print(f"\nParams: LANE_WIDTH={LANE_WIDTH}m, D={D}m, v0={v0}m/s")
    print(f"MPC: Ts={mpc.Ts}s, P={mpc.P}, M={mpc.M}, δ_max={math.degrees(mpc.delta_max):.1f}°")
    print(f"Solver: {'scipy SLSQP' if HAS_SCIPY else 'numpy PGD fallback'}\n")

    # Run MPC in a loop simulating the maneuver
    dt = mpc.Ts
    s = 0.0
    y, theta = 0.0, 0.0

    print(f"{'step':>4}  {'s(m)':>6}  {'y(m)':>7}  {'y_ref':>7}  {'θ(°)':>6}  "
          f"{'δ(°)':>6}  {'v(m/s)':>6}  {'ms':>5}")
    print("-" * 70)

    for step in range(80):
        z0 = np.array([y, theta])
        delta_rad, v_cmd = mpc.solve(z0, s, D, LANE_WIDTH, +1.0, v0, v_ref, u_prev)

        y_ref = mpc.quintic_ref(s, D, LANE_WIDTH, +1.0)

        print(f"{step:4d}  {s:6.3f}  {y:7.4f}  {y_ref:7.4f}  "
              f"{math.degrees(theta):6.2f}  {math.degrees(delta_rad):6.2f}  "
              f"{v_cmd:6.3f}  {mpc.last_solve_ms:5.1f}")

        # Simulate forward (kinematic bicycle, Euler integration)
        v0 = v_cmd
        y += v0 * theta * dt
        theta += (v0 * delta_rad / mpc.L) * dt
        s += v0 * dt
        u_prev = delta_rad

    print(f"\nFinal  y = {y:.4f} m   (target = {LANE_WIDTH:.4f} m)")
    print(f"Error    = {abs(y - LANE_WIDTH) * 1000:.1f} mm")
    print(f"Avg solve time = {mpc.last_solve_ms:.1f} ms")