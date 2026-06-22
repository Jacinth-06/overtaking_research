# Linear MPC Controller Implementation Walkthrough

I've completed the implementation of the Linear Model Predictive Controller (MPC) for the JetRacer's overtaking and recovery maneuvers in `version5.py`. Here's a breakdown of the new system architecture and the key technical decisions.

## 1. Kinematic Bicycle Model Formulation
The core of the MPC relies on a discrete-time linearised kinematic bicycle model. Because the Jetson Nano needs to run the entire control loop at ~20Hz (50ms sample time) alongside the vision and LiDAR pipelines, a full nonlinear solver was deemed too computationally expensive.

> [!NOTE]
> The model is linearized around the current speed ($v_0$) and a zero steering angle. This allows the problem to be framed as a convex Quadratic Program (QP), which can be solved in single-digit milliseconds.

### State Space Matrices
The continuous dynamics were linearised and discretized (exact zero-order hold) into the following matrices:
```math
A_d = \begin{bmatrix} 1 & v_0 \cdot T_s \\ 0 & 1 \end{bmatrix}, \quad B_d = \begin{bmatrix} \frac{v_0^2 \cdot T_s^2}{2L} \\ \frac{v_0 \cdot T_s}{L} \end{bmatrix}
```
Where:
*   $T_s = 0.05$ s (20 Hz)
*   $L = 0.15$ m (wheelbase)

## 2. Speed Control Integration
Per your request, the MPC now actively controls **both steering and speed**.

**How it works:**
*   The MPC takes a target speed reference ($v_{ref}$), which is snapshotted from the slider's `enc_speed` at the moment the maneuver begins.
*   The solver calculates an optimal speed command ($v_{cmd}$) alongside the optimal steering angle ($\delta_{cmd}$).
*   It balances tracking the target speed against minimizing aggressive accelerations (via an $R_v$ penalty weight on speed deviations).
*   In `version5.py`, the returned $v_{cmd}$ is mapped proportionally back to a `throttle_cmd` for the JetRacer's motor driver, ensuring the car maintains its momentum through the curve without exceeding hardware limits.

## 3. Hard and Soft Constraints
The transition to MPC allows us to strictly enforce the physical limitations of the JetRacer:

### Hard Constraints (Inputs)
*   **Steering:** $|\delta_k| \le 25^\circ$. The solver will *never* command an angle outside the steering rack's physical limits.
*   **Speed:** $v_{min} \le v_k \le v_{max}$. Keeps the command within a sensible band ($0.05$ to $0.30$ m/s).

### Soft Constraints (Outputs)
*   **Lane Boundaries:** $y_{min} - \epsilon \le \hat{y}_k \le y_{max} + \epsilon$.
*   To prevent the solver from failing if the car is bumped slightly outside the lane boundaries, the lateral constraints are "soft." They use a slack variable ($\epsilon \ge 0$) with a heavy penalty ($\rho = 1000$) in the cost function. The solver will violate the boundary only if it absolutely must to satisfy the hard steering constraints.

## 4. Solver Implementation (`mpc_controller.py`)
I built `LaneChangeMPC` as a self-contained module so it can be tested independently of the JetRacer hardware.

> [!TIP]
> **Numpy Fallback Solver:** The primary solver uses `scipy.optimize.minimize(method='SLSQP')`. However, knowing that Jetson environments can sometimes have tricky `scipy` installations, I implemented a pure-`numpy` Projected Gradient Descent (PGD) fallback. If `scipy` isn't found at runtime, the controller will automatically seamlessly downgrade to the numpy solver.

## 5. Integration into `version5.py`
The state machine in `control_loop` was updated to seamlessly hand over control from the vision PID to the MPC.

```python
# From version5.py
elif autonomy_state == "OVERTAKING":
    # ...
    # ── MPC solve (rate-limited to MPC_INTERVAL) ──────────────
    if now - mpc_last_time >= MPC_INTERVAL:
        z0 = np.array([pos_y, yaw])
        delta_rad, v_cmd = mpc.solve(
            z0, s, D, LANE_WIDTH, +1.0,
            max(enc_speed, 0.01), mpc_speed_ref, mpc_prev_delta
        )
        mpc_last_delta = delta_rad
        mpc_last_v = v_cmd
        mpc_last_time = now
```

*   **Rate Limiting:** The MPC solve is rate-limited to its design frequency ($T_s = 50$ms). If the camera loop runs faster, it holds the previous command, preventing instability from variable sample times.
*   **Telemetry:** Added `mpc_solve_ms`, `mpc_delta_deg`, and `mpc_v_cmd` to the telemetry payload so you can graph the MPC's performance and execution time in the dashboard.

## Next Steps
The new `tests/version5.py` is ready to be deployed to the Jetson Nano for live testing. You can run `python3 tests/test_mpc.py` on the Jetson to run the offline unit tests before trying it on the track.
