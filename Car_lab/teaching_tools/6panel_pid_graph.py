#!/usr/bin/env python3
"""
Extended PID Controller Visualization – Complete Curve Navigation Cycle
----------------------------------------------------------------------
This script generates a six‑panel figure that explains how P, I, D and the
combined PID signal evolve as a small robot negotiates a right‑hand curve.
Changes compared with the original draft:

* **Derivative term is now calculated per definition**  D = Kd * (de/dt).
  – Key‑frame derivative uses the video‑frame interval (0.1 s).
  – Dense derivative uses numpy.gradient and is optionally smoothed by a
    zero‑lag Savitzky–Golay filter (scipy).  If SciPy is absent the code
    runs without smoothing.
* Removed the manual derivative hack and its artefacts (flat purple walls).
* Axis limits auto‑scale except where hard limits aid interpretation.
* Internal constants grouped near the top for quick edits.
* Prints a compact verification table that you can paste into LaTeX.

Written for Python ≥ 3.8.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ---------------------------------------------------------------------------
# User‑tunable parameters
# ---------------------------------------------------------------------------
Kp: float = 0.8       # degrees / pixel
Ki: float = 0.1       # degrees / pixel·s
Kd: float = 0.3       # degrees / pixel·s

frame_rate: float = 10.0          # camera FPS   → Δt = 0.1 s
wheelbase: float  = 0.20          # m (robot geometry)
speed: float      = 1.0           # m/s (assumed constant)

# Smoothing (set to 0 to disable)
SG_WINDOW: int = 9     # must be odd and ≥ polyorder + 2
SG_POLY: int   = 2

# Output
FIG_NAME = "2extended_pid_visualization.png"
FIG_DPI  = 300

# ---------------------------------------------------------------------------
# Key‑frame scenario (6 frames: N‑2 … N+3)
# ---------------------------------------------------------------------------
key_times  = np.arange(0.0, 0.6, 0.1)                # [0.0 … 0.5] s
key_errors = np.array([5.0, 15.0, 25.0, 15.0, 5.0, 0.0])  # px
frame_dt   = 1.0 / frame_rate                        # 0.1 s

# ---------------------------------------------------------------------------
# Helper: create smooth error profile
# ---------------------------------------------------------------------------

def make_error_profile(times: np.ndarray,
                        key_t: np.ndarray,
                        key_e: np.ndarray) -> np.ndarray:
    """Piece‑wise linear interpolation through key points."""
    return np.interp(times, key_t, key_e)


# ---------------------------------------------------------------------------
# Dense timeline (for nice curves)
# ---------------------------------------------------------------------------
extended_times  = np.linspace(-0.05, 0.65, 150)        # s
extended_errors = make_error_profile(extended_times, key_times, key_errors)

dt_dense = np.gradient(extended_times)                 # variable step (≈0.0047 s)

# ---------------------------------------------------------------------------
# P term (straightforward)
# ---------------------------------------------------------------------------
P_terms      = Kp * extended_errors
key_P_terms  = Kp * key_errors

# ---------------------------------------------------------------------------
# I term (rectangular integration, causal)
# ---------------------------------------------------------------------------
# I_terms = np.zeros_like(extended_errors)
# int_sum = 0.0
# for i in range(1, len(extended_errors)):
#     int_sum += extended_errors[i] * dt_dense[i]
#     I_terms[i] = Ki * int_sum
# key_I_terms = I_terms[np.searchsorted(extended_times, key_times)]
I_terms = np.zeros_like(extended_errors)
int_sum = 0.0
for i in range(1, len(extended_errors)):
    int_sum += extended_errors[i] * dt_dense[i]
    I_terms[i] = Ki * int_sum

# Key frame integral (cumulative sum of discrete errors)
key_integral_sums = np.cumsum(key_errors)  # [5, 20, 45, 60, 65, 65]
key_I_terms = Ki * key_integral_sums       # Ki * cumulative sum

# ---------------------------------------------------------------------------
# D term –– correct formulation
# ---------------------------------------------------------------------------
# 1) Key‑frame derivative -----------------------------------------------------
prev_error = 3.0  # px, assumed error one frame *before* N‑2
error_series = np.insert(key_errors, 0, prev_error)
error_changes = np.diff(error_series) / frame_dt      # px / s
key_D_terms   = Kd * error_changes                    # deg

# 2) Dense derivative ---------------------------------------------------------
D_terms = Kd * np.gradient(extended_errors, extended_times)

# Optional Savitzky–Golay smoothing (keeps zero phase, causal enough here)
try:
    if SG_WINDOW > 2:
        from scipy.signal import savgol_filter
        D_terms = savgol_filter(D_terms, window_length=SG_WINDOW, polyorder=SG_POLY)
except ModuleNotFoundError:
    pass  # SciPy not installed – carry on without smoothing

# ---------------------------------------------------------------------------
# Combined control signal
# ---------------------------------------------------------------------------
I_terms_interpolated = np.interp(extended_times, key_times, key_I_terms)
Total_terms = P_terms + I_terms_interpolated + D_terms
key_totals  = key_P_terms + key_I_terms + key_D_terms

# ---------------------------------------------------------------------------
# Robot kinematics (bicycle model, tiny sideways slip for realism)
# ---------------------------------------------------------------------------

def simulate_robot(times: np.ndarray,
                   steering_deg: np.ndarray,
                   *,
                   v: float = speed,
                   L: float = wheelbase) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) trajectory in metres."""
    x = np.linspace(0, 6, len(times))         # forward distance ~ track length
    y = np.zeros_like(x)
    heading = np.zeros_like(x)

    for i in range(1, len(x)):
        dt = times[i] - times[i - 1]
        steer_rad = np.radians(steering_deg[i])
        if abs(steer_rad) > 1e-3:
            R = L / np.tan(steer_rad)
            omega = v / R
        else:
            omega = 0.0
        heading[i] = heading[i - 1] + omega * dt
        y[i] = y[i - 1] + v * np.sin(heading[i]) * dt * 0.1  # damped lateral slip
    return x, y

robot_x, robot_y = simulate_robot(extended_times, Total_terms)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
plt.style.use("default")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

frame_labels = ["N-2", "N-1", "N", "N+1", "N+2", "N+3"]

# 1) Error signal -------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(extended_times, extended_errors, "b-", lw=3)
ax1.scatter(key_times, key_errors, c="red", s=80, zorder=5)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Error (px)")
ax1.set_title("1. Error Signal e(t)")
ax1.grid(alpha=0.3)

for t, e, lbl in zip(key_times, key_errors, frame_labels):
    ax1.annotate(f"{lbl}\n{e:.0f}px", xy=(t, e), xytext=(t + 0.01, e + 2),
                 textcoords="data", arrowprops=dict(arrowstyle="->", color="red"),
                 ha="center", fontsize=9, fontweight="bold")

# 2) P term -------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(extended_times, P_terms, "r-", lw=3)
ax2.scatter(key_times, key_P_terms, c="darkred", s=80)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Control (deg)")
ax2.set_title(f"2. Proportional: P = {Kp} × e(t)")
ax2.grid(alpha=0.3)

# 3. INTEGRAL TERM - Enhanced visualization
ax3 = fig.add_subplot(gs[0, 2])

# Plot error curve for reference
ax3.plot(extended_times, extended_errors, 'b--', linewidth=2, alpha=0.6, label='Error e(t)')

# Create progressive hatched areas for integration
time_segments = [
    (extended_times <= key_times[1]),
    (extended_times <= key_times[2]) & (extended_times > key_times[1]),
    (extended_times <= key_times[3]) & (extended_times > key_times[2]),
    (extended_times <= key_times[4]) & (extended_times > key_times[3]),
    (extended_times <= key_times[5]) & (extended_times > key_times[4])
]
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
alphas = [0.4, 0.5, 0.6, 0.4, 0.3]

for i, (mask, color, alpha) in enumerate(zip(time_segments, colors, alphas)):
    if np.any(mask):
        times_seg = extended_times[mask]
        errors_seg = extended_errors[mask]
        ax3.fill_between(times_seg, 0, errors_seg, color=color, alpha=alpha, 
                        hatch='///', edgecolor='black', linewidth=0.5)

# Plot integral value curve
ax3_twin = ax3.twinx()
ax3_twin.plot(extended_times, I_terms, 'g-', linewidth=4, label='Integral Value')
ax3_twin.scatter(key_times, key_I_terms, color='darkgreen', s=100, zorder=5)
ax3_twin.set_ylabel('Integral Term (degrees)', color='green')
ax3_twin.tick_params(axis='y', labelcolor='green')

ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Error (pixels)', color='blue')
ax3.set_title(f'3. Integral: I = {Ki} × ∫e(τ)dτ', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-2, 30)
ax3_twin.set_ylim(0, 8)

# Annotate integral progression
for i, (t, i_val) in enumerate(zip(key_times, key_I_terms)):
    if i % 2 == 0:  # Every other point
        ax3_twin.annotate(f'{i_val:.1f}°', xy=(t, i_val), xytext=(t+0.01, i_val+0.3),
                         arrowprops=dict(arrowstyle='->', color='darkgreen'),
                         fontsize=9, ha='center', fontweight='bold')

# 4) D term -------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(extended_times, D_terms, color="purple", lw=3)
ax4.scatter(key_times, key_D_terms, c="indigo", s=80)
ax4.axhline(0, color="black", lw=0.8, alpha=0.4)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Control (deg)")
ax4.set_title(f"4. Derivative: D = {Kd} × de/dt")
ax4.grid(alpha=0.3)


# 5) Combined PID -------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(extended_times, P_terms, "r-", lw=2, alpha=0.6, label="P")
ax5.plot(extended_times, I_terms_interpolated, "g-", lw=2, alpha=0.6, label="I")
ax5.plot(extended_times, D_terms, color="purple", lw=2, alpha=0.6, label="D")
ax5.plot(extended_times, Total_terms, "k-", lw=3, label="Total PID")
ax5.scatter(key_times, key_totals, c="orange", s=100, zorder=5)
ax5.axhline(30, color="red", ls="--", alpha=0.7, label="±30° limit")
ax5.axhline(-30, color="red", ls="--", alpha=0.7)
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Control (deg)")
ax5.set_title("5. Combined PID Output")
ax5.grid(alpha=0.3)
ax5.legend(fontsize=9)

# 6) Robot response -----------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
line_x = robot_x
line_y = np.ones_like(line_x) * 1.5
curve_mask = (line_x >= 2.0) & (line_x <= 4.0)
line_y[curve_mask] += 0.3 * np.sin(np.pi * (line_x[curve_mask] - 2.0) / 2.0)

ax6.plot(line_x, line_y, "k--", lw=3, label="Target Line")
ax6.plot(robot_x, robot_y + 1.5, "r-", lw=3, label="Robot Path")
for i, t in enumerate(key_times):
    idx = np.searchsorted(robot_x, 6 * t / extended_times[-1])
    if i % 2 == 0:
        ax6.scatter(robot_x[idx], robot_y[idx] + 1.5, s=120, c="orange", zorder=5)
        ax6.annotate(f"Frame {frame_labels[i]}",
                     xy=(robot_x[idx], robot_y[idx] + 1.5),
                     xytext=(robot_x[idx], robot_y[idx] + 1.8),
                     arrowprops=dict(arrowstyle="->", color="orange"),
                     ha="center", fontsize=9, fontweight="bold")
ax6.set_xlabel("Distance (m)")
ax6.set_ylabel("Lateral position (m)")
ax6.set_title("6. Robot Response – Complete Curve Navigation")
ax6.set_ylim(1.0, 2.0)
ax6.grid(alpha=0.3)
ax6.legend()

fig.suptitle("Extended PID Line Following: Complete Curve Navigation Cycle",
             fontsize=16, fontweight="bold", y=0.95)
plt.tight_layout()

# Save figure
plt.savefig(FIG_NAME, dpi=FIG_DPI, bbox_inches="tight")
print(f"Figure saved as {FIG_NAME}")

# ---------------------------------------------------------------------------
# Verification table (key frames)
# ---------------------------------------------------------------------------
print("\nKey‑frame PID calculations:")
print("Frame   t(s)  e(px)   P(°)  I(°)   D(°)   Total(°)")
print("———   ————  —————  —————  —————  —————  ——————")
for lbl, t, e, p, i_val, d, tot in zip(frame_labels, key_times,
                                       key_errors, key_P_terms,
                                       key_I_terms, key_D_terms,
                                       key_totals):
    print(f"{lbl:5s}  {t:+.1f}  {e:+5.1f}  {p:+6.1f}  {i_val:+6.1f}  {d:+6.1f}  {tot:+7.1f}")
