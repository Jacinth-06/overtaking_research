"""
plot_pid_vs_mpc.py
==================
Conference-paper quality figures comparing PID (version3) and MPC (version5)
autonomous overtake runs logged from your RC car telemetry.

Produces 5 figures saved to ./figures/:
  Fig 1 – Lateral Position (lateral_pos_vs_distance.pdf)
  Fig 2 – Steering Angle vs Time (steering_angle_vs_time.pdf)
  Fig 3 – Lidar / State Timeline (lidar_state_timeline.pdf)
  Fig 4 – Lane Centering Error – PID FOLLOW baseline (lane_error_follow.pdf)
  Fig 5 – MPC Solve Time Histogram (mpc_solve_time_hist.pdf)

Usage:
  pip install matplotlib numpy scipy
  python plot_pid_vs_mpc.py

Step-by-step guide is embedded in section comments below.
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.ndimage import uniform_filter1d  # for lightweight smoothing

# ─────────────────────────────────────────────
# 0.  GLOBAL STYLE  (IEEE / conference standard)
# ─────────────────────────────────────────────
# Use a clean, white-background style close to what IEEE expects.
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "lines.linewidth":   1.5,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour palette – colourblind-friendly (Wong 2011)
C_PID  = "#E69F00"   # amber
C_MPC  = "#0072B2"   # blue
C_LIDAR= "#CC79A7"   # mauve
C_ERR  = "#009E73"   # green

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD  DATA
# ─────────────────────────────────────────────
# STEP 1 ▸ Load both JSON telemetry files.
# Keys are millisecond-epoch strings; values are dicts of sensor readings.
print("Loading telemetry …")
with open("mpc_static_overtake.json") as f:
    mpc_raw = json.load(f)
with open("pid_static_overtake.json") as f:
    pid_raw = json.load(f)

def parse_log(raw: dict) -> dict:
    """Convert timestamp-keyed dict → arrays sorted by time."""
    rows = sorted(raw.values(), key=lambda r: float(r["timestamp"]))
    t0   = float(rows[0]["timestamp"])
    out  = {k: [] for k in rows[0].keys()}
    out["t"] = []
    for r in rows:
        out["t"].append(float(r["timestamp"]) - t0)
        for k, v in r.items():
            out[k].append(v)
    return {k: np.array(v) if isinstance(v[0], (int, float, bool)) else v
            for k, v in out.items()}

mpc = parse_log(mpc_raw)
pid = parse_log(pid_raw)

print(f"  MPC  : {len(mpc['t']):.0f} samples  |  duration {mpc['t'][-1]:.1f} s")
print(f"  PID  : {len(pid['t']):.0f} samples  |  duration {pid['t'][-1]:.1f} s")

# ─────────────────────────────────────────────
# 2.  DERIVE  LATERAL  POSITION  FROM  ENCODER
# ─────────────────────────────────────────────
# STEP 2 ▸ enc_dist gives cumulative odometry distance (m).
# 'error' is the fractional lane error from vision (−1…+1).
# Lateral position ≈ error × (lane_width_px / nominal_px_per_m).
# Since nominal_lane_width is logged as 0 we use lane_width directly
# as a *relative* lateral offset (pixels, centred = 0).
# For physical units, multiply by a calibration factor k_lat (m/px).
# This is clearly labelled on the axis so reviewers know.

def lateral_pos(log):
    """Signed lateral offset in pixels (positive = right of centre)."""
    return log["error"].astype(float) * log["lane_width"].astype(float) / 2.0

pid_lat = lateral_pos(pid)
mpc_lat = lateral_pos(mpc)

# Cumulative distance (enc_dist is already in metres per tick)
def cum_dist(log):
    d = np.cumsum(np.abs(np.diff(log["enc_dist"].astype(float),
                                  prepend=log["enc_dist"][0])))
    return d

pid_dist = cum_dist(pid)
mpc_dist = cum_dist(mpc)

# ─────────────────────────────────────────────
#  HELPER: state colour bands
# ─────────────────────────────────────────────
STATE_COLOURS = {
    "FOLLOW":   "#AED6F1",   # light blue
    "APPROACH": "#FAD7A0",   # light orange
    "OVERTAKE": "#A9DFBF",   # light green
    "RETURN":   "#D2B4DE",   # light purple
    "STOP":     "#F9EBEA",   # light red
}

def draw_state_bands(ax, t, states, alpha=0.25, label=True):
    """Shade background by autonomy_state."""
    if not isinstance(states, (list,)):
        states = list(states)
    prev_state = states[0]
    t_start    = t[0]
    drawn      = set()
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            colour = STATE_COLOURS.get(prev_state, "#DDDDDD")
            ax.axvspan(t_start, t[i], color=colour, alpha=alpha,
                       label=prev_state if (label and prev_state not in drawn) else "_nolegend_")
            drawn.add(prev_state)
            prev_state = states[i]
            t_start    = t[i]

# ═══════════════════════════════════════════════
#  FIG 1 – LATERAL POSITION vs DISTANCE TRAVELLED
# ═══════════════════════════════════════════════
# STEP 3 ▸ Core result: both trajectories on one axes.
# Smooth with a small window to reduce per-frame noise while keeping
# the overshoot clearly visible (window chosen conservatively).

SMOOTH_WIN = 7   # samples (~0.1 s at 60 Hz)

fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))

pid_lat_sm = uniform_filter1d(pid_lat, size=SMOOTH_WIN)
mpc_lat_sm = uniform_filter1d(mpc_lat, size=SMOOTH_WIN)

ax1.plot(pid_dist, pid_lat_sm, color=C_PID, label="PID – Quintic v3",
         linestyle="-")
ax1.plot(mpc_dist, mpc_lat_sm, color=C_MPC, label="MPC – v5",
         linestyle="--")
ax1.axhline(0, color="black", linewidth=0.8, linestyle=":", label="Lane centre")

ax1.set_xlabel("Distance Travelled (m)")
ax1.set_ylabel("Lateral Offset (px · lane-width⁻¹)")
ax1.set_title("Fig. 1 – Lateral Position vs Distance: PID vs MPC")
ax1.legend(loc="upper right")

# Annotate max overshoot for PID
peak_idx = np.argmax(np.abs(pid_lat_sm))
ax1.annotate(f"PID overshoot\n{pid_lat_sm[peak_idx]:.0f} px",
             xy=(pid_dist[peak_idx], pid_lat_sm[peak_idx]),
             xytext=(pid_dist[peak_idx] + 0.2, pid_lat_sm[peak_idx] * 1.15),
             arrowprops=dict(arrowstyle="->", color=C_PID, lw=1.0),
             fontsize=8, color=C_PID)

fig1.tight_layout()
fig1.savefig("figures/lateral_pos_vs_distance.pdf")
fig1.savefig("figures/lateral_pos_vs_distance.png")
print("  ✓ Fig 1 saved")

# ═══════════════════════════════════════════════
#  FIG 2 – STEERING ANGLE vs TIME
# ═══════════════════════════════════════════════
# STEP 4 ▸ PID uses 'steer' (−1…+1 normalised).
# MPC logs 'mpc_delta_deg' directly in degrees.
# Convert PID steer to degrees using max servo angle (assume ±30°).
MAX_STEER_DEG = 30.0

pid_steer_deg = pid["steer"].astype(float) * MAX_STEER_DEG
mpc_steer_deg = mpc["mpc_delta_deg"].astype(float)

fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))

draw_state_bands(ax2, mpc["t"], mpc["autonomy_state"])

ax2.plot(pid["t"], uniform_filter1d(pid_steer_deg, size=SMOOTH_WIN),
         color=C_PID, label=f"PID steer × {MAX_STEER_DEG}°", alpha=0.9)
ax2.plot(mpc["t"], uniform_filter1d(mpc_steer_deg, size=SMOOTH_WIN),
         color=C_MPC, label="MPC δ (deg)", linestyle="--", alpha=0.9)
ax2.axhline(0, color="black", linewidth=0.7, linestyle=":")

# Draw ±MAX line
for sign in (+1, -1):
    ax2.axhline(sign * MAX_STEER_DEG, color="grey", linewidth=0.7,
                linestyle=":", alpha=0.6)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Steering Angle (°)")
ax2.set_title("Fig. 2 – Steering Angle vs Time")

# State legend patches
state_patches = [mpatches.Patch(color=c, alpha=0.4, label=s)
                 for s, c in STATE_COLOURS.items()
                 if s in set(mpc["autonomy_state"])]
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + state_patches,
           labels=labels + [p.get_label() for p in state_patches],
           ncol=2, fontsize=8, loc="upper right")

fig2.tight_layout()
fig2.savefig("figures/steering_angle_vs_time.pdf")
fig2.savefig("figures/steering_angle_vs_time.png")
print("  ✓ Fig 2 saved")

# ═══════════════════════════════════════════════
#  FIG 3 – LIDAR TRIGGER & STATE TIMELINE (MPC run)
# ═══════════════════════════════════════════════
# STEP 5 ▸ Three sub-axes on the same time axis.
# Row A: lidar_closest (mm) – obstacle proximity
# Row B: autonomy_state colour map
# Row C: steer command overlaid

fig3, axes = plt.subplots(3, 1, figsize=(6.5, 5.0),
                           sharex=True,
                           gridspec_kw={"height_ratios": [2.5, 0.8, 1.5]})
fig3.suptitle("Fig. 3 – Lidar Trigger & State Timeline (MPC run)", y=1.01)

t_m = mpc["t"]
lid = mpc["lidar_closest"].astype(float)
stop_dist = mpc["stop_distance"].astype(float)

# -- Row A: lidar
ax3a = axes[0]
ax3a.fill_between(t_m, lid, alpha=0.15, color=C_LIDAR)
ax3a.plot(t_m, lid, color=C_LIDAR, linewidth=1.0, label="lidar_closest")
ax3a.plot(t_m, stop_dist, color="red", linewidth=0.9, linestyle="--",
          label="stop_distance threshold")
ax3a.set_ylabel("Lidar (mm)")
ax3a.legend(fontsize=8, loc="upper right")

# Annotate minimum lidar reading
min_idx = np.argmin(lid)
ax3a.annotate(f"{lid[min_idx]:.0f} mm",
              xy=(t_m[min_idx], lid[min_idx]),
              xytext=(t_m[min_idx] + 1, lid[min_idx] + 100),
              arrowprops=dict(arrowstyle="->", lw=0.8),
              fontsize=8)

# -- Row B: state as colour bands
ax3b = axes[1]
ax3b.set_yticks([])
ax3b.set_ylabel("State", rotation=0, labelpad=30, va="center")
states_m = list(mpc["autonomy_state"])
prev = states_m[0]; t_start = t_m[0]
for i in range(1, len(states_m)):
    if states_m[i] != prev or i == len(states_m) - 1:
        colour = STATE_COLOURS.get(prev, "#DDDDDD")
        ax3b.axvspan(t_start, t_m[i], color=colour, alpha=0.8)
        mid = (t_start + t_m[i]) / 2
        ax3b.text(mid, 0.5, prev, ha="center", va="center",
                  fontsize=7, transform=ax3b.get_xaxis_transform())
        prev = states_m[i]; t_start = t_m[i]

# -- Row C: steer command
ax3c = axes[2]
ax3c.plot(t_m, uniform_filter1d(mpc_steer_deg, size=SMOOTH_WIN),
          color=C_MPC, linewidth=1.2, label="MPC δ (°)")
ax3c.axhline(0, color="black", linewidth=0.6, linestyle=":")
ax3c.set_ylabel("Steer (°)")
ax3c.set_xlabel("Time (s)")
ax3c.legend(fontsize=8, loc="upper right")

fig3.tight_layout()
fig3.savefig("figures/lidar_state_timeline.pdf")
fig3.savefig("figures/lidar_state_timeline.png")
print("  ✓ Fig 3 saved")

# ═══════════════════════════════════════════════
#  FIG 4 – LANE CENTERING ERROR DURING FOLLOW (PID)
# ═══════════════════════════════════════════════
# STEP 6 ▸ Isolate FOLLOW segments from the PID log.
# 'error' is the vision fractional error (0 = perfectly centred).

follow_mask = np.array(pid["autonomy_state"]) == "FOLLOW"
t_f   = pid["t"][follow_mask]
err_f = pid["error"][follow_mask].astype(float)

mae  = np.mean(np.abs(err_f))
std_ = np.std(err_f)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(7, 3.0),
                                   gridspec_kw={"width_ratios": [3, 1]})
fig4.suptitle("Fig. 4 – Lane Centering Error: PID FOLLOW Baseline")

# Time-series
ax4a.plot(t_f, err_f, color=C_ERR, linewidth=0.8, alpha=0.7, label="Error")
ax4a.axhline(0,    color="black",   linewidth=0.8, linestyle=":")
ax4a.axhline(mae,  color=C_PID,     linewidth=1.0, linestyle="--",
             label=f"MAE = {mae:.4f}")
ax4a.axhline(-mae, color=C_PID,     linewidth=1.0, linestyle="--")
ax4a.fill_between(t_f, mae, -mae, alpha=0.08, color=C_PID)
ax4a.set_xlabel("Time (s)")
ax4a.set_ylabel("Lane Error (fractional)")
ax4a.legend(fontsize=8)

# Histogram
ax4b.hist(err_f, bins=40, orientation="horizontal",
          color=C_ERR, alpha=0.7, edgecolor="white", linewidth=0.3)
ax4b.axhline(mae,  color=C_PID, linewidth=1.0, linestyle="--")
ax4b.axhline(-mae, color=C_PID, linewidth=1.0, linestyle="--")
ax4b.axhline(0,    color="black", linewidth=0.8, linestyle=":")
ax4b.set_xlabel("Count")
ax4b.set_yticks([])
ax4b.set_title(f"σ = {std_:.4f}", fontsize=9)

# Stats box
textstr = f"MAE = {mae:.4f}\nσ   = {std_:.4f}\nn   = {len(err_f)}"
ax4a.text(0.02, 0.97, textstr, transform=ax4a.transAxes,
          fontsize=8, verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

fig4.tight_layout()
fig4.savefig("figures/lane_error_follow.pdf")
fig4.savefig("figures/lane_error_follow.png")
print("  ✓ Fig 4 saved")

# ═══════════════════════════════════════════════
#  FIG 5 – MPC SOLVE TIME HISTOGRAM
# ═══════════════════════════════════════════════
# STEP 7 ▸ mpc_solve_ms is already logged in milliseconds.
# Budget line at 50 ms (= 20 Hz).

solve_ms = mpc["mpc_solve_ms"].astype(float)
solve_ms = solve_ms[solve_ms > 0]   # drop zeroes (non-active ticks)

BUDGET_MS  = 50.0
p50 = np.percentile(solve_ms, 50)
p95 = np.percentile(solve_ms, 95)
p99 = np.percentile(solve_ms, 99)
n_over_budget = np.sum(solve_ms > BUDGET_MS)

fig5, ax5 = plt.subplots(figsize=(5.0, 3.2))

counts, edges, patches = ax5.hist(solve_ms, bins=60, color=C_MPC,
                                   alpha=0.75, edgecolor="white", linewidth=0.3)

# Colour bins that exceed budget red
for patch, left in zip(patches, edges[:-1]):
    if left >= BUDGET_MS:
        patch.set_facecolor("#D9534F")

ax5.axvline(BUDGET_MS, color="red",   linewidth=1.5, linestyle="--",
            label=f"Budget {BUDGET_MS:.0f} ms (20 Hz)")
ax5.axvline(p50,       color="black", linewidth=1.2, linestyle="-",
            label=f"P50 = {p50:.1f} ms")
ax5.axvline(p95,       color=C_PID,   linewidth=1.2, linestyle="-.",
            label=f"P95 = {p95:.1f} ms")
ax5.axvline(p99,       color=C_LIDAR, linewidth=1.2, linestyle=":",
            label=f"P99 = {p99:.1f} ms")

ax5.set_xlabel("Solve Time (ms)")
ax5.set_ylabel("Count")
ax5.set_title("Fig. 5 – MPC Solve Time Histogram (Jetson Nano)")
ax5.legend(fontsize=8)

# Annotation
textstr = (f"n        = {len(solve_ms)}\n"
           f"Mean   = {np.mean(solve_ms):.2f} ms\n"
           f"Max    = {np.max(solve_ms):.2f} ms\n"
           f"> budget = {n_over_budget} ({100*n_over_budget/len(solve_ms):.1f}%)")
ax5.text(0.97, 0.97, textstr, transform=ax5.transAxes,
         fontsize=7.5, verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig5.tight_layout()
fig5.savefig("figures/mpc_solve_time_hist.pdf")
fig5.savefig("figures/mpc_solve_time_hist.png")
print("  ✓ Fig 5 saved")

# ─────────────────────────────────────────────
# SUMMARY STATS  (paste into your paper's table)
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY STATISTICS FOR PAPER TABLE")
print("="*55)
print(f"PID  – Lane error MAE  : {mae:.5f}  σ={std_:.5f}")
print(f"PID  – Peak steer      : {np.max(np.abs(pid_steer_deg)):.2f}°")
print(f"MPC  – Peak steer      : {np.max(np.abs(mpc_steer_deg)):.2f}°")
print(f"MPC  – Steer σ         : {np.std(mpc_steer_deg):.3f}°")
print(f"MPC  – Solve P50/P95/P99: {p50:.1f}/{p95:.1f}/{p99:.1f} ms")
print(f"MPC  – % over 50 ms    : {100*n_over_budget/len(solve_ms):.2f}%")
print(f"PID  – Lateral peak    : {np.max(np.abs(pid_lat_sm)):.1f} px")
print(f"MPC  – Lateral peak    : {np.max(np.abs(mpc_lat_sm)):.1f} px")
print("="*55)
print("\nAll figures written to ./figures/  (PDF + PNG)")
