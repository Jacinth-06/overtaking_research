# MPC Controller for Overtaking & Recovery in version5.py

Replace the PID trajectory controller in the OVERTAKING and RECOVERY state machine phases with a Linear MPC (Model Predictive Control) controller. The FOLLOW state (vision lane-centering PID) remains unchanged.

## User Review Required

> [!IMPORTANT]
> **Maximum steering angle**: version4.py uses `MAX_STEER_DEG = 25`. I will use **δ_max = 25°** as the physical steering limit. Is this correct for your JetRacer?

> [!IMPORTANT]
> **Solver dependency**: The MPC needs a QP solver. I plan to use **scipy.optimize.minimize (SLSQP)** which should already be on the Jetson (`scipy` comes with jetpack's numpy ecosystem). If scipy is not available, I'll implement a pure-numpy projected gradient descent solver. **Is scipy installed on your Jetson Nano?**

> [!IMPORTANT]
> **`car.steer()` mapping**: I assume `car.steer(1.0)` = full right lock = +δ_max, and `car.steer(-1.0)` = full left lock = -δ_max. This is a linear mapping: `steer_cmd = δ / δ_max`. Is this correct?

## Open Questions

> [!WARNING]
> **Speed during maneuver**: Currently `s_copy["speed"]` (default 0.15) is sent as constant throttle during overtaking. The MPC treats velocity `v` as a **measured disturbance** (known but not controlled). Should the MPC also control speed, or keep it constant like the PID version?

> [!WARNING]
> **IMU-based state estimation**: version5.py integrates gyro → yaw → lateral position. Version4.py abandoned this because of IMU drift. The MPC can work either way:
> - **Option A**: Use IMU-based `[y, θ]` as measured state (same as current version5.py) — MPC's receding horizon partially compensates for drift.
> - **Option B**: Use encoder-only dead-reckoning (distance traveled `s` from encoder, heading from bicycle model prediction) — no IMU noise, but open-loop for heading.
> I recommend **Option A** since MPC's feedback nature handles drift better than PID did.

---

## Complete Mathematical Formulation

### 1. Vehicle Model — Kinematic Bicycle (no slip)

Given:
- **Wheelbase** `L = 0.15 m`
- **Width** `W = 0.18 m` (not used in kinematics, but relevant for lane constraints)
- **No tyre grip/slip** → pure kinematic model

The continuous-time kinematic bicycle model with state measured relative to maneuver start:

```
State:  z = [y, θ]ᵀ
            y = lateral position (meters, relative to maneuver start)
            θ = heading angle (radians, relative to straight-ahead)

Input:  u = δ  (front wheel steering angle, radians)

Dynamics:
    ẏ = v · sin(θ) ≈ v · θ    (small angle, valid for gentle lane changes)
    θ̇ = v · tan(δ) / L ≈ v · δ / L    (small angle)
```

> [!NOTE]
> Small angle linearisation is valid here because:
> - Lane change of ~0.112m over 0.30m gives max heading ≈ 20° ≈ 0.35 rad
> - sin(0.35) = 0.34, θ = 0.35 → error ~3%, acceptable for MPC since it re-solves every step

### 2. Continuous State-Space (Linearised)

```
ż = Ac · z + Bc · u

Ac = [0   v]      Bc = [ 0  ]
     [0   0]           [v/L ]
```

### 3. Discretisation (Zero-Order Hold at Ts)

Using exact matrix exponential for this simple system:

```
Ad = [1   v·Ts]      Bd = [v·Ts²/(2L)]
     [0     1 ]           [  v·Ts/L   ]
```

**Chosen parameters** (from your spec):
| Parameter | Symbol | Value | Rationale |
|---|---|---|---|
| Sample time | Ts | 0.05 s (20 Hz) | Matches lidar rate, achievable on Jetson |
| Prediction horizon | P | 25 steps (= 1.25 s) | 25 × Ts, within your 20-30 × Ts range |
| Control horizon | M | 4 steps (= 0.2 s) | 16% of P, within your 10-20% range |
| Max steering | δ_max | 0.4363 rad (25°) | From version4.py |

### 4. Prediction Model — Condensed Form

Over the prediction horizon, the future states are expressed as a function of the current state `z₀` and future inputs `U = [u₀, u₁, ..., u_{M-1}]ᵀ`:

```
z₁ = Ad·z₀ + Bd·u₀
z₂ = Ad²·z₀ + Ad·Bd·u₀ + Bd·u₁
⋮
zₖ = Ad^k·z₀ + Σ_{j=0}^{k-1} Ad^{k-1-j}·Bd·u_j
```

For k ≥ M, the input is held: `uₖ = u_{M-1}` (control horizon constraint).

In matrix form:

```
Z = Ψ·z₀ + Θ·U

where:
    Z = [z₁, z₂, ..., zₚ]ᵀ         (2P × 1)
    U = [u₀, u₁, ..., u_{M-1}]ᵀ     (M × 1)
    Ψ = [Ad; Ad²; ...; Ad^P]         (2P × 2)  — free response matrix
    Θ = forced response matrix        (2P × M)  — built from Ad^i·Bd products
```

We only track `y` (first element of each `zₖ`), so we extract via:

```
Y = C·Z      where C = [1 0] applied to each block → Y is (P × 1)
```

### 5. Reference Trajectory

The reference `y_ref` at each future step k is computed from the **quintic polynomial + kick blend** (same as current version5.py):

```python
def quintic_ref(s, D, LANE_WIDTH, direction):
    """
    s = encoder distance from maneuver start at future step k
    D = OVERTAKE_MANEUVER_DIST
    direction = +1 (overtake left) or -1 (recovery right)
    """
    if s < D:
        r = s / D
        poly = 10*r³ - 15*r⁴ + 6*r⁵
        kick = r^0.5 · (1 - r)
        poly = poly + 0.4 · kick
        return direction * LANE_WIDTH * poly
    else:
        return direction * LANE_WIDTH
```

At each MPC solve, for step k in [1..P]:
```
s_future_k = s_current + v · k · Ts
y_ref_k = quintic_ref(s_future_k, D, LANE_WIDTH, direction)
```

### 6. Cost Function

```
J = (Y - Y_ref)ᵀ · Q̄ · (Y - Y_ref)   ← tracking error (output)
  + ΔUᵀ · R̄ · ΔU                       ← input rate penalty (smoothness)
  + ρ · ε²                              ← soft output constraint penalty

where:
    Q̄ = diag(q, q, ..., q)    (P × P)     q = 100.0  (heavy tracking weight)
    R̄ = diag(r, r, ..., r)    (M × M)     r = 1.0    (mild rate penalty)
    ΔU = [u₀ - u_{-1}, u₁ - u₀, ..., u_{M-1} - u_{M-2}]  (input increments)
    ρ = 1000.0                              (soft constraint penalty)
    ε ≥ 0                                   (slack variable)
```

Substituting `Y = C_y · (Ψ·z₀ + Θ·U)`:

```
J = (C_y·Θ·U + C_y·Ψ·z₀ - Y_ref)ᵀ · Q̄ · (C_y·Θ·U + C_y·Ψ·z₀ - Y_ref)
  + (T·U + t₀)ᵀ · R̄ · (T·U + t₀)
  + ρ · ε²

where T is the differencing matrix for ΔU and t₀ = [-u_{prev}; 0; ...; 0]
```

This is a standard **Quadratic Program (QP)**:

```
min  ½ Uᵀ·H·U + fᵀ·U + const
 U

H = Θ_yᵀ·Q̄·Θ_y + Tᵀ·R̄·T           (M × M, positive definite)
f = Θ_yᵀ·Q̄·(Ψ_y·z₀ - Y_ref) + Tᵀ·R̄·t₀
```

### 7. Constraints

**Hard input constraints** (steering angle limits):

```
-δ_max ≤ uₖ ≤ +δ_max    for k = 0, ..., M-1

In normalised steer_cmd units:  -1.0 ≤ steer_cmd ≤ +1.0
In radians:                     -0.4363 ≤ δ ≤ +0.4363
```

**Soft output constraints** (lateral position bounds):

```
y_min - ε ≤ yₖ ≤ y_max + ε    for k = 1, ..., P
ε ≥ 0

where:
    y_min = -0.05 m     (small negative overshoot allowed)
    y_max = LANE_WIDTH + 0.05 m   (small positive overshoot allowed)
```

These are enforced as **soft constraints** with penalty ρ·ε² in the cost. The slack ε allows temporary violation (e.g., overshoot during transient) without making the QP infeasible.

### 8. MPC Algorithm (per control loop iteration)

```
1. Read current state: z₀ = [pos_y, yaw]  (from IMU integration)
2. Read current speed: v = enc_speed
3. Read encoder distance: s_now = enc_dist - start_enc_dist
4. Compute reference trajectory Y_ref for steps k=1..P
5. Build Ad, Bd (depend on v which may change)
6. Build prediction matrices Ψ, Θ
7. Form QP: H, f, bounds
8. Solve QP → U* = [u₀*, u₁*, ..., u_{M-1}*]
9. Apply ONLY first element: δ = u₀*
10. Convert: steer_cmd = δ / δ_max, clamp to [-1, 1]
11. car.steer(steer_cmd)
```

### 9. Computational Budget

For Jetson Nano at 20 Hz (50 ms per cycle):
- QP size: M = 4 decision variables, P = 25 constraints
- Using SLSQP: typical solve time ~0.5-2 ms for this tiny problem
- Well within the 50 ms budget (camera processing takes ~15-20 ms)

---

## Proposed Changes

### MPC Controller Module

#### [NEW] [mpc_controller.py](file:///Users/jacinth/Development/overtaking_research/tests/mpc_controller.py)

Self-contained MPC solver class:
- `class LaneChangeMPC` — encapsulates all MPC math
- `__init__(Ts, P, M, L, delta_max, ...)` — build constant matrices
- `update_speed(v)` — rebuild Ad, Bd, Ψ, Θ when speed changes
- `solve(z0, s_now, D, lane_width, direction, u_prev)` — solve QP, return `steer_cmd`
- `quintic_ref(s, D, lane_width, direction)` — reference trajectory generator
- Pure numpy + scipy, no external dependencies

---

### Control Loop (version5.py)

#### [MODIFY] [version5.py](file:///Users/jacinth/Development/overtaking_research/tests/version5.py)

Changes to `control_loop()`:

1. **Import** `LaneChangeMPC` from `mpc_controller.py`
2. **Instantiate** `mpc = LaneChangeMPC(...)` at top of control_loop, alongside existing constants
3. **OVERTAKING state**: Replace the PID block (lines 682-694) with:
   ```python
   z0 = np.array([pos_y, yaw])
   steer_cmd = mpc.solve(z0, s, D, LANE_WIDTH, +1.0, last_mpc_u)
   last_mpc_u = steer_cmd * delta_max  # store in radians for next solve
   ```
4. **RECOVERY state**: Replace the PID block (lines 727-737) with the same MPC call but `direction = -1.0`
5. **State transitions** remain identical (lidar gating, distance checks, etc.)
6. **Remove** `pid_traj` dict and `TRAJ_KP/KI/KD` constants (no longer needed)
7. **Add** `mpc_solve_time` to telemetry for monitoring MPC computation time

The vision PID in FOLLOW state is **NOT modified**.

---

## Verification Plan

### Automated Tests
- Create `tests/test_mpc.py` — unit test the MPC solver offline:
  ```bash
  python tests/test_mpc.py
  ```
  - Test 1: Straight-line (zero reference) → MPC outputs ~0 steering
  - Test 2: Step reference (instant lane change) → MPC outputs smooth steering profile
  - Test 3: Quintic reference → MPC tracks within 1cm after settling
  - Test 4: Timing benchmark → solve < 5ms on laptop (< 10ms on Jetson)
  - Test 5: Hard constraints → output never exceeds δ_max

### Manual Verification
- Deploy to Jetson, run `python tests/version5.py`
- Monitor dashboard: MPC solve time in telemetry
- Test overtake maneuver with obstacle → verify smooth lateral transition
- Test recovery maneuver → verify return to original lane
- Compare MPC telemetry vs PID telemetry (lateral tracking error, steering smoothness)
