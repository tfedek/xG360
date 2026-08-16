"""
09_spatial_feature_documentation.py  –  NEW (professor review, point P8)
=========================================================================
Documents the spatial (360) feature construction with:
  1. A freeze-frame example visualization (one shot, showing all
     visible players, shot line, and pressure zone)
  2. A sensitivity analysis for the pressure_score radius parameter
     (default: 10 m, constant 0.5)
  3. A pseudocode block printed to stdout (for inclusion in paper appendix)

[P8] Per reviewer: "The open_goal_angle_ratio and pressure_score
construction should be validated through a freeze-frame example
and sensitivity analysis."
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    GOAL_X, GOAL_Y, GOAL_WIDTH,
    LEFT_POST_Y, RIGHT_POST_Y, BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX,
)
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "spatial_documentation"
ensure_dirs(OUT_DIR)


# ============================================================
# 1. Pseudocode (printed, for paper appendix)
# ============================================================
PSEUDOCODE = """
PSEUDOCODE: Spatial (360) Feature Construction
================================================

INPUT: freeze_frame  — list of {actor_position: (x, y), actor: {teammate: bool}}
       shot_location — (x_s, y_s)
       goal_center   — (120.0, 40.0)
       goal_posts    — left: (120.0, 36.34), right: (120.0, 43.66)

--- open_goal_angle_ratio ---
  total_angle = angle subtended by goalposts from shot_location
  blocked_angle = 0
  FOR each opponent in freeze_frame:
      IF opponent is between shooter and goal (x > x_s):
          blocked_width = effective blocking width (e.g., 0.6 m shoulder)
          add to blocked_angle proportionally
  open_goal_angle_ratio = max(0, 1 − blocked_angle / total_angle)

--- pressure_score ---
  pressure_score = 0
  FOR each opponent in freeze_frame:
      d = Euclidean distance from opponent to shot_location
      IF d < RADIUS (default: 10 m / ~10.94 yd):
          pressure_score += 1 / (d + EPSILON)   (EPSILON = 0.5 to avoid div-by-zero)

--- nearest_defender_to_shot_line ---
  shot_line = vector from shot_location to goal_center
  FOR each opponent in freeze_frame:
      perp_distance = perpendicular distance from opponent to shot_line
  nearest_defender_to_shot_line = min(perp_distance)

OUTPUT: open_goal_angle_ratio  (float, 0–1)
        pressure_score         (float, ≥0)
        nearest_defender_to_shot_line (float, meters)
"""

print(PSEUDOCODE)
with open(OUT_DIR / "pseudocode_spatial_features.txt", "w") as f:
    f.write(PSEUDOCODE)


# ============================================================
# 2. Sensitivity analysis: pressure_score radius and epsilon
# ============================================================
def pressure_score(opponents_xy, shot_xy, radius, epsilon):
    """Compute pressure score for a given radius and epsilon."""
    score = 0.0
    for ox, oy in opponents_xy:
        d = np.hypot(ox - shot_xy[0], oy - shot_xy[1])
        if d < radius:
            score += 1.0 / (d + epsilon)
    return score


# Simulate a typical shot scenario: 4 opponents at varying distances
shot = np.array([105.0, 40.0])
opponents = np.array([
    [108.0, 39.0],   # ~3m, very close
    [110.0, 41.0],   # ~5m
    [103.0, 37.0],   # ~4m, behind shooter
    [112.0, 43.0],   # ~8m
])

# Grid search over radius and epsilon
radii   = [5, 7, 10, 12, 15]
epsilons = [0.1, 0.3, 0.5, 1.0, 2.0]

rows = []
for r in radii:
    for eps in epsilons:
        ps = pressure_score(opponents, shot, r, eps)
        rows.append({"radius_m": r, "epsilon": eps, "pressure_score": round(ps, 4)})

sensitivity_df = pd.DataFrame(rows)
sensitivity_df.to_csv(OUT_DIR / "pressure_score_sensitivity.csv", index=False)

# Pivot for display
pivot = sensitivity_df.pivot(index="radius_m", columns="epsilon", values="pressure_score")
print("\n[P8] pressure_score sensitivity (rows=radius, cols=epsilon):")
print(pivot.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
for eps in epsilons:
    vals = [pressure_score(opponents, shot, r, eps) for r in radii]
    ax.plot(radii, vals, marker="o", label=f"ε={eps}")
ax.axvline(x=10, linestyle="--", color="gray", label="default radius=10m")
ax.set_xlabel("Radius (m)")
ax.set_ylabel("pressure_score")
ax.set_title("[P8] pressure_score sensitivity to radius and epsilon")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "pressure_score_sensitivity.png", dpi=160)
plt.close()
print(f"Saved: {OUT_DIR / 'pressure_score_sensitivity.png'}")


# ============================================================
# 3. Freeze-frame example visualization
# ============================================================
# Fabricate a representative freeze frame (real freeze frames
# would require loading from the StatsBomb raw JSON, which is
# not included in the portable package — this example uses
# synthetic but geometrically plausible positions)

fig, ax = plt.subplots(figsize=(10, 6))

# Pitch outline (half-pitch, StatsBomb coords)
ax.set_xlim(90, 122)
ax.set_ylim(10, 70)
ax.set_facecolor("#2d7a2d")
ax.add_patch(patches.Rectangle((90, 10), 32, 60, linewidth=2, edgecolor="white", facecolor="none"))
ax.add_patch(patches.Rectangle((BOX_X_MIN, BOX_Y_MIN), 120-BOX_X_MIN, BOX_Y_MAX-BOX_Y_MIN,
                                 linewidth=1.5, edgecolor="white", facecolor="none"))
# Goal
ax.add_patch(patches.Rectangle((120, LEFT_POST_Y), 2, GOAL_WIDTH,
                                 linewidth=2, edgecolor="white", facecolor="none"))

# Shot location
sx, sy = 106.0, 38.0
ax.scatter(sx, sy, s=200, color="yellow", zorder=5, label="Shooter")

# Teammates
teammates = [(104, 45), (108, 50)]
for tx, ty in teammates:
    ax.scatter(tx, ty, s=120, color="blue", zorder=4)
ax.scatter([], [], color="blue", label="Teammates")

# Opponents
opp_positions = [(109, 39), (112, 41), (111, 36), (108, 44)]
for ox, oy in opp_positions:
    ax.scatter(ox, oy, s=120, color="red", zorder=4)
ax.scatter([], [], color="red", label="Opponents")

# Goalkeeper
ax.scatter(119, 40, s=150, color="orange", marker="D", zorder=5, label="Goalkeeper")

# Shot line
ax.plot([sx, GOAL_X], [sy, GOAL_Y], "y--", linewidth=1.5, label="Shot line to goal center")

# Pressure radius circle
circle = plt.Circle((sx, sy), 10 / (105/68 * 1), color="yellow", fill=False,
                      linestyle=":", linewidth=1.5, label="Pressure radius (10m)")
ax.add_patch(circle)

# Goal angle lines
ax.plot([sx, GOAL_X], [sy, LEFT_POST_Y], "w-", linewidth=1, alpha=0.5)
ax.plot([sx, GOAL_X], [sy, RIGHT_POST_Y], "w-", linewidth=1, alpha=0.5)

ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
ax.set_title("[P8] Freeze-frame example: spatial feature illustration\n"
              "(synthetic positions, geometrically representative)")
ax.set_xlabel("x (StatsBomb, 0–120)")
ax.set_ylabel("y (StatsBomb, 0–80)")
plt.tight_layout()
plt.savefig(OUT_DIR / "freeze_frame_example.png", dpi=180)
plt.close()
print(f"Saved: {OUT_DIR / 'freeze_frame_example.png'}")
print(f"\nAll P8 outputs saved in: {OUT_DIR}")
