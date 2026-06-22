#!/usr/bin/env python3
"""
test_mpc.py — Offline unit tests for the MPC lane-change controller.

Run:   python test_mpc.py
       (no JetRacer hardware needed)
"""

import sys
import os
import time
import math
import numpy as np

# Ensure tests/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_controller import LaneChangeMPC


def test_straight_line():
    """MPC with zero reference should output near-zero steering."""
    print("TEST 1: Straight line (zero reference) ... ", end="", flush=True)
    mpc = LaneChangeMPC()
    z0 = np.array([0.0, 0.0])
    # s_now > D means maneuver is done, reference = +LANE_WIDTH (a constant)
    # Instead, set D very large so reference stays near 0
    delta, v = mpc.solve(z0, s_now=0.0, D=100.0, lane_width=0.112,
                         direction=+1.0, v0=0.08, v_ref=0.08, u_prev_delta=0.0)
    # At s=0, the quintic ref is 0, so MPC should barely steer
    assert abs(delta) < 0.05, f"Expected near-zero δ, got {math.degrees(delta):.2f}°"
    print(f"PASS  (δ = {math.degrees(delta):.3f}°, v = {v:.3f} m/s)")


def test_quintic_tracking():
    """Simulate MPC tracking the full quintic overtake trajectory."""
    print("TEST 2: Quintic tracking (overtake simulation) ... ", flush=True)
    mpc = LaneChangeMPC(
        Ts=0.05, P=25, M=4, L=0.15, delta_max_deg=25.0,
        q_y=100.0, r_delta=1.0, r_v=10.0, rho=1000.0,
        v_min=0.05, v_max=0.30,
    )
    LANE_WIDTH = 0.112
    D = 0.30
    v0 = 0.08
    dt = mpc.Ts

    y, theta, s = 0.0, 0.0, 0.0
    u_prev = 0.0
    max_error = 0.0

    for step in range(100):
        z0 = np.array([y, theta])
        delta_rad, v_cmd = mpc.solve(z0, s, D, LANE_WIDTH, +1.0, v0, 0.08, u_prev)
        u_prev = delta_rad
        v0 = v_cmd

        # Simulate
        y += v0 * theta * dt
        theta += (v0 * delta_rad / mpc.L) * dt
        s += v0 * dt

        y_ref = mpc.quintic_ref(s, D, LANE_WIDTH, +1.0)
        err = abs(y - y_ref)
        max_error = max(max_error, err)

        if s > D + 0.3:
            break

    final_ref = LANE_WIDTH
    final_err = abs(y - final_ref)
    print(f"  Final y={y:.4f}m  ref={final_ref:.4f}m  err={final_err*1000:.1f}mm  "
          f"max_track_err={max_error*1000:.1f}mm")
    assert final_err < 0.020, f"Final error {final_err*1000:.1f}mm > 20mm"
    print("  PASS")


def test_recovery_direction():
    """MPC with direction=-1 should steer in the opposite direction."""
    print("TEST 3: Recovery direction (-1) ... ", end="", flush=True)
    mpc = LaneChangeMPC()
    z0 = np.array([0.0, 0.0])
    delta_pos, _ = mpc.solve(z0, 0.05, 0.30, 0.112, +1.0, 0.08, 0.08, 0.0)
    delta_neg, _ = mpc.solve(z0, 0.05, 0.30, 0.112, -1.0, 0.08, 0.08, 0.0)
    # They should be opposite in sign
    assert delta_pos * delta_neg <= 0, (
        f"Expected opposite signs: +dir={math.degrees(delta_pos):.2f}°, "
        f"-dir={math.degrees(delta_neg):.2f}°")
    print(f"PASS  (+dir={math.degrees(delta_pos):.2f}°, -dir={math.degrees(delta_neg):.2f}°)")


def test_hard_constraints():
    """MPC output must never exceed δ_max or speed bounds."""
    print("TEST 4: Hard constraint satisfaction ... ", end="", flush=True)
    mpc = LaneChangeMPC(delta_max_deg=25.0, v_min=0.05, v_max=0.30)

    # Extreme initial condition — far from reference
    for y0 in [-0.5, 0.0, 0.5]:
        for theta0 in [-0.3, 0.0, 0.3]:
            z0 = np.array([y0, theta0])
            delta, v = mpc.solve(z0, 0.1, 0.30, 0.112, +1.0, 0.08, 0.08, 0.0)
            assert abs(delta) <= mpc.delta_max + 1e-6, (
                f"δ={math.degrees(delta):.2f}° exceeds limit {25}°")
            assert v >= mpc.v_min - 1e-6 and v <= mpc.v_max + 1e-6, (
                f"v={v:.3f} outside [{mpc.v_min}, {mpc.v_max}]")
    print("PASS")


def test_timing():
    """MPC solve should be fast enough for real-time (< 10ms target)."""
    print("TEST 5: Timing benchmark ... ", end="", flush=True)
    mpc = LaneChangeMPC()
    z0 = np.array([0.0, 0.0])

    times = []
    for _ in range(50):
        t0 = time.time()
        mpc.solve(z0, 0.05, 0.30, 0.112, +1.0, 0.08, 0.08, 0.0)
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    mx = max(times)
    print(f"avg={avg:.1f}ms  max={mx:.1f}ms  ", end="")
    if avg < 10:
        print("PASS")
    else:
        print(f"WARN (avg > 10ms, may be slow on Jetson)")


def test_speed_control():
    """MPC should output speed near v_ref when no disturbance."""
    print("TEST 6: Speed control ... ", end="", flush=True)
    mpc = LaneChangeMPC(r_v=50.0)  # higher weight on speed tracking
    z0 = np.array([0.0, 0.0])
    v_ref = 0.12
    _, v_cmd = mpc.solve(z0, 0.0, 100.0, 0.112, +1.0, 0.12, v_ref, 0.0)
    err = abs(v_cmd - v_ref)
    assert err < 0.05, f"Speed error {err:.3f} m/s > 50mm/s"
    print(f"PASS  (v_cmd={v_cmd:.3f}, v_ref={v_ref:.3f}, err={err*1000:.0f}mm/s)")


if __name__ == "__main__":
    print("=" * 60)
    print("MPC Controller — Unit Tests")
    print("=" * 60)
    print()

    test_straight_line()
    test_quintic_tracking()
    test_recovery_direction()
    test_hard_constraints()
    test_timing()
    test_speed_control()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
