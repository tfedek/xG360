from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from football_xg.config import (
    MATCHES_CSV, EVENTS_DIR, THREE_SIXTY_DIR, PROCESSED_DIR, OUTPUT_DIR,
    GOAL_X, GOAL_Y, LEFT_POST_Y, RIGHT_POST_Y, GOAL_WIDTH
)
from football_xg.data_utils import load_json, name, safe_loc, ensure_dirs
from football_xg.geometry import (
    shot_distance, shot_angle, euclidean, point_in_triangle,
    estimated_blocked_angle, distance_point_to_segment, in_box
)

ensure_dirs(PROCESSED_DIR, OUTPUT_DIR)


def extract_shots(events, match_meta):
    rows = []

    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue

        shot = ev.get("shot", {})
        x, y = safe_loc(ev)

        if pd.isna(x) or pd.isna(y):
            continue

        outcome = name(shot.get("outcome"))

        rows.append({
            "id": ev.get("id"),
            "match_id": int(match_meta["match_id"]),
            "tournament": match_meta["tournament"],
            "competition_id": int(match_meta["competition_id"]),
            "season_id": int(match_meta["season_id"]),
            "match_date": match_meta.get("match_date"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "period": ev.get("period"),
            "team": name(ev.get("team")),
            "player": name(ev.get("player")),
            "x": x,
            "y": y,
            "distance": shot_distance(x, y),
            "angle": shot_angle(x, y),
            "shot_body_part": name(shot.get("body_part")),
            "shot_type": name(shot.get("type")),
            "shot_technique": name(shot.get("technique")),
            "shot_first_time": bool(shot.get("first_time", False)),
            "shot_one_on_one": bool(shot.get("one_on_one", False)),
            "shot_open_goal": bool(shot.get("open_goal", False)),
            "shot_statsbomb_xg": shot.get("statsbomb_xg"),
            "shot_outcome": outcome,
            "goal": 1 if outcome == "Goal" else 0,
        })

    return rows


def extract_360_features(frames, shots_by_id):
    rows = []

    for frame_event in frames:
        event_id = frame_event.get("event_uuid")

        if event_id not in shots_by_id:
            continue

        shot = shots_by_id[event_id]
        sx = float(shot["x"])
        sy = float(shot["y"])

        freeze = frame_event.get("freeze_frame", [])
        visible_area = frame_event.get("visible_area", [])

        teammates = []
        opponents = []
        keepers = []

        for p in freeze:
            loc = p.get("location")

            if not isinstance(loc, list) or len(loc) < 2:
                continue

            px = float(loc[0])
            py = float(loc[1])
            tup = (px, py)

            if p.get("keeper") is True:
                keepers.append(tup)
            elif p.get("teammate") is True:
                teammates.append(tup)
            else:
                opponents.append(tup)

                # Fallback: some files include position metadata.
                if p.get("position", {}).get("name") == "Goalkeeper":
                    keepers.append(tup)

        opponent_distances = [
            euclidean(sx, sy, px, py)
            for px, py in opponents
        ]

        teammate_distances = [
            euclidean(sx, sy, px, py)
            for px, py in teammates
        ]

        # ------------------------------------------------------------
        # Old cone-based features: useful but too strict.
        # ------------------------------------------------------------
        defenders_in_cone = 0
        blocked_angle_sum = 0.0

        for px, py in opponents:
            inside_cone = point_in_triangle(
                px, py,
                sx, sy,
                GOAL_X, LEFT_POST_Y,
                GOAL_X, RIGHT_POST_Y
            )

            if inside_cone:
                defenders_in_cone += 1
                blocked_angle_sum += estimated_blocked_angle(px, py, sx, sy)

        base_angle = shot_angle(sx, sy)

        if pd.isna(base_angle) or base_angle <= 0:
            open_angle = np.nan
            visible_goal_ratio = np.nan
        else:
            open_angle = max(base_angle - blocked_angle_sum, 0)
            visible_goal_ratio = open_angle / base_angle

        opponents_between = [
            (px, py)
            for px, py in opponents
            if px >= sx and point_in_triangle(
                px, py,
                sx, sy,
                GOAL_X, LEFT_POST_Y,
                GOAL_X, RIGHT_POST_Y
            )
        ]

        # ------------------------------------------------------------
        # NEW shot-line features:
        # More realistic than narrow triangle only.
        # Distance from each defender to direct line shooter -> goal centre.
        # ------------------------------------------------------------
        shot_line_distances = []

        for px, py in opponents:
            # only defenders between shooter and goal direction
            if px >= sx:
                d_line = distance_point_to_segment(
                    px, py,
                    sx, sy,
                    GOAL_X, GOAL_Y
                )
                shot_line_distances.append(d_line)

        nearest_defender_to_shot_line = (
            min(shot_line_distances) if shot_line_distances else np.nan
        )

        defenders_within_1m_of_shot_line = sum(
            d <= 1 for d in shot_line_distances
        )

        defenders_within_2m_of_shot_line = sum(
            d <= 2 for d in shot_line_distances
        )

        # ------------------------------------------------------------
        # Goalkeeper features
        # ------------------------------------------------------------
        if keepers:
            keeper = min(
                keepers,
                key=lambda k: euclidean(k[0], k[1], GOAL_X, GOAL_Y)
            )
            kx, ky = keeper

            goalkeeper_distance = euclidean(sx, sy, kx, ky)
            goalkeeper_distance_to_goal = euclidean(kx, ky, GOAL_X, GOAL_Y)
            goalkeeper_in_cone = int(
                point_in_triangle(
                    kx, ky,
                    sx, sy,
                    GOAL_X, LEFT_POST_Y,
                    GOAL_X, RIGHT_POST_Y
                )
            )
            goalkeeper_offset_from_goal_center = abs(ky - GOAL_Y)
        else:
            goalkeeper_distance = np.nan
            goalkeeper_distance_to_goal = np.nan
            goalkeeper_in_cone = np.nan
            goalkeeper_offset_from_goal_center = np.nan

        # ------------------------------------------------------------
        # General spatial features
        # ------------------------------------------------------------
        opp_x = [p[0] for p in opponents]
        opp_y = [p[1] for p in opponents]

        nearest_defender_distance = (
            min(opponent_distances) if opponent_distances else np.nan
        )

        second_nearest_defender_distance = (
            sorted(opponent_distances)[1]
            if len(opponent_distances) > 1 else np.nan
        )

        mean_defender_distance = (
            float(np.mean(opponent_distances)) if opponent_distances else np.nan
        )

        std_defender_distance = (
            float(np.std(opponent_distances)) if opponent_distances else np.nan
        )

        nearest_teammate_distance = (
            min(teammate_distances) if teammate_distances else np.nan
        )

        pressure_score = sum(
            1 / (d + 0.5)
            for d in opponent_distances
            if d <= 10
        )

        local_opponent_density_5m = (
            sum(d <= 5 for d in opponent_distances) / (np.pi * 5 * 5)
        )

        local_opponent_density_10m = (
            sum(d <= 10 for d in opponent_distances) / (np.pi * 10 * 10)
        )

        local_teammate_density_10m = (
            sum(d <= 10 for d in teammate_distances) / (np.pi * 10 * 10)
        )

        opponents_in_box = sum(
            in_box(px, py)
            for px, py in opponents
        )

        teammates_in_box = sum(
            in_box(px, py)
            for px, py in teammates
        )

        teammates_ahead_of_ball = sum(
            px > sx
            for px, py in teammates
        )

        shot_cone_density = (
            defenders_in_cone / max(base_angle, 1e-6)
            if not pd.isna(base_angle) else np.nan
        )

        shooting_corridor_area = 0.5 * GOAL_WIDTH * max(GOAL_X - sx, 0)

        shooting_corridor_width = (
            GOAL_WIDTH * max(GOAL_X - sx, 0)
            / max(shot_distance(sx, sy), 1e-6)
        )

        rows.append({
            "id": event_id,

            "num_teammates_360": len(teammates),
            "num_opponents_360": len(opponents),

            "goalkeeper_distance_360": goalkeeper_distance,
            "goalkeeper_distance_to_goal_360": goalkeeper_distance_to_goal,
            "goalkeeper_in_shot_cone_360": goalkeeper_in_cone,
            "goalkeeper_offset_from_goal_center_360": goalkeeper_offset_from_goal_center,

            "nearest_defender_distance_360": nearest_defender_distance,
            "second_nearest_defender_distance_360": second_nearest_defender_distance,
            "mean_defender_distance_360": mean_defender_distance,
            "std_defender_distance_360": std_defender_distance,
            "nearest_teammate_distance_360": nearest_teammate_distance,

            "defenders_within_2m_360": sum(d <= 2 for d in opponent_distances),
            "defenders_within_5m_360": sum(d <= 5 for d in opponent_distances),
            "defenders_within_10m_360": sum(d <= 10 for d in opponent_distances),

            "teammates_within_5m_360": sum(d <= 5 for d in teammate_distances),
            "teammates_within_10m_360": sum(d <= 10 for d in teammate_distances),

            "defenders_in_shot_cone_360": defenders_in_cone,
            "opponents_between_shooter_and_goal_360": len(opponents_between),

            "nearest_defender_to_shot_line_360": nearest_defender_to_shot_line,
            "defenders_within_1m_of_shot_line_360": defenders_within_1m_of_shot_line,
            "defenders_within_2m_of_shot_line_360": defenders_within_2m_of_shot_line,

            "shot_cone_density_360": shot_cone_density,
            "blocked_angle_360": blocked_angle_sum,
            "open_angle_360": open_angle,
            "open_angle_ratio_360": visible_goal_ratio,
            "visible_goal_ratio_360": visible_goal_ratio,

            "free_space_radius_360": nearest_defender_distance,
            "pressure_score_360": pressure_score,

            "local_opponent_density_5m_360": local_opponent_density_5m,
            "local_opponent_density_10m_360": local_opponent_density_10m,
            "local_teammate_density_10m_360": local_teammate_density_10m,

            "opponents_in_box_360": opponents_in_box,
            "teammates_in_box_360": teammates_in_box,
            "teammates_ahead_of_ball_360": teammates_ahead_of_ball,

            "defensive_line_avg_x_360": float(np.mean(opp_x)) if opp_x else np.nan,
            "defensive_line_width_y_360": (
                float(np.max(opp_y) - np.min(opp_y))
                if len(opp_y) > 1 else np.nan
            ),

            "shooting_corridor_width_360": shooting_corridor_width,
            "shooting_corridor_area_360": shooting_corridor_area,

            "visible_area_points_count_360": (
                len(visible_area) if isinstance(visible_area, list) else 0
            ),
        })

    return rows


def build_dataset():
    matches = pd.read_csv(MATCHES_CSV)

    all_shots = []
    all_360 = []

    for _, match in tqdm(matches.iterrows(), total=len(matches), desc="Building dataset with shot-line features"):
        match_id = int(match["match_id"])

        event_path = EVENTS_DIR / f"{match_id}.json"
        frame_path = THREE_SIXTY_DIR / f"{match_id}.json"

        if not event_path.exists():
            continue

        events = load_json(event_path)
        shots = extract_shots(events, match.to_dict())

        if not shots:
            continue

        all_shots.extend(shots)

        if frame_path.exists():
            frames = load_json(frame_path)
            shots_by_id = {s["id"]: s for s in shots}
            all_360.extend(extract_360_features(frames, shots_by_id))

    shots_df = pd.DataFrame(all_shots)
    f360_df = pd.DataFrame(all_360)

    if f360_df.empty:
        return shots_df

    return shots_df.merge(f360_df, on="id", how="left")


def run_checks(df):
    print("\n=== DATASET INFO ===")
    print(df.shape)

    print("\n=== GOAL DISTRIBUTION ===")
    print(df["goal"].value_counts())
    print(df["goal"].value_counts(normalize=True))

    print("\n=== TOURNAMENT SUMMARY ===")
    summary = df.groupby("tournament")["goal"].agg(["count", "sum", "mean"])
    print(summary)
    summary.to_csv(OUTPUT_DIR / "tournament_summary_shotline.csv")

    missing = df.isna().mean().sort_values(ascending=False)

    print("\n=== MISSING VALUES TOP 30 ===")
    print(missing.head(30))
    missing.to_csv(OUTPUT_DIR / "missing_values_shotline.csv")

    plt.figure(figsize=(8, 5))
    df["distance"].hist(bins=40)
    plt.title("Shot Distance Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distance_distribution_shotline.png", dpi=150)
    plt.close()

    # VIF diagnostic only; not used directly for XGBoost.
    vif_cols = [
        c for c in df.columns
        if c.endswith("_360")
    ] + ["distance", "angle"]

    vif_cols = [
        c for c in vif_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[vif_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.nunique() > 1]
    X_const = sm.add_constant(X)

    rows = []

    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue

        try:
            vif = variance_inflation_factor(X_const.values, i)
        except Exception:
            vif = np.nan

        rows.append({
            "feature": col,
            "VIF": vif,
        })

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(OUTPUT_DIR / "vif_results_shotline.csv", index=False)

    print("\n=== VIF TOP 25 ===")
    print(vif_df.head(25))


def main():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(
            f"Missing {MATCHES_CSV}. Run 01_download_data.py first."
        )

    df = build_dataset()

    output_file = PROCESSED_DIR / "shots_with_360_dataset.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSaved dataset: {output_file}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    run_checks(df)

    print("\nDONE.")


if __name__ == "__main__":
    main()
