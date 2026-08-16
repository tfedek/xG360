"""
export_for_web_v2.py
Export v2 pipeline models and data to JSON format for web app.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from football_xg.config import (
    DATASET_PATH, MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL, TARGET,
    RAW_DIR, THREE_SIXTY_DIR, EVENTS_DIR
)
from football_xg.data_utils import load_json

OUTPUT_DIR = Path("/Users/fedektom/Downloads/xg_analiza/web/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} shots, {df[TARGET].sum()} goals")

    print("Loading v2 models...")
    model_a = joblib.load(Path.home() / "data/models/model_a_classic_logistic.pkl")
    model_b = joblib.load(Path.home() / "data/models/model_b_360_v3_logistic.pkl")

    # Predictions for all shots
    feats_a = MODEL_A_NUMERIC + CATEGORICAL
    feats_b = MODEL_B_NUMERIC + CATEGORICAL

    print("Generating predictions...")
    X_a = df[feats_a].copy()
    X_b = df[feats_b].copy()
    # Fill NaN for 360 features (missing = no data, use median as safe default)
    X_b = X_b.fillna(X_b.median(numeric_only=True))
    
    df["xg_a"] = model_a.predict_proba(X_a)[:, 1]
    df["xg_b"] = model_b.predict_proba(X_b)[:, 1]

    # Tournament labels
    tournament_labels = {
        "FIFA World Cup 2022": "Svetsko prvenstvo 2022",
        "UEFA Euro 2020": "Evropsko prvenstvo 2020",
        "UEFA Euro 2024": "Evropsko prvenstvo 2024",
    }

    # Export shots.json
    print("Exporting shots.json...")
    shots_out = []
    for _, row in df.iterrows():
        is_penalty = row.get("shot_type") == "Penalty"
        shots_out.append({
            "event_id": row["id"],
            "match_id": row["match_id"],
            "tournament": row["tournament"],
            "tournament_label": tournament_labels.get(row["tournament"], row["tournament"]),
            "team": row["team"],
            "player": row["player"],
            "minute": int(row["minute"]),
            "period": int(row["period"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "distance_to_goal": float(row["distance"]),
            "shot_angle_deg": float(row["angle"]),
            "body_part": row["shot_body_part"],
            "shot_type": row["shot_type"],
            "is_penalty": is_penalty,
            "is_goal": bool(row[TARGET]),
            "statsbomb_xg": float(row["shot_statsbomb_xg"]) if pd.notna(row.get("shot_statsbomb_xg")) else None,
            "xg_a": float(row["xg_a"]),
            "xg_b": float(row["xg_b"]),
            # 360 features for display
            "n_defenders_in_cone": int(row["defenders_in_shot_cone_360"]) if pd.notna(row.get("defenders_in_shot_cone_360")) else None,
            "pressure_score": float(row["pressure_score_360"]) if pd.notna(row.get("pressure_score_360")) else None,
            "nearest_defender_to_shot_line": float(row["nearest_defender_to_shot_line_360"]) if pd.notna(row.get("nearest_defender_to_shot_line_360")) else None,
            "goalkeeper_distance": float(row["goalkeeper_distance_360"]) if pd.notna(row.get("goalkeeper_distance_360")) else None,
        })

    with open(OUTPUT_DIR / "shots.json", "w") as f:
        json.dump(shots_out, f)
    print(f"  {len(shots_out)} shots exported")

    # Export freeze_frames.json from raw data
    print("Exporting freeze_frames.json...")
    frames_out = {}
    events_dir = EVENTS_DIR
    if events_dir.exists():
        for event_file in sorted(events_dir.iterdir()):
            if not event_file.suffix == ".json":
                continue
            events = load_json(event_file)
            for ev in events:
                if ev.get("type", {}).get("name") != "Shot":
                    continue
                ev_id = ev.get("id")
                ff = ev.get("shot", {}).get("freeze_frame")
                if ff and ev_id:
                    frame = []
                    for p in ff:
                        loc = p.get("location", [None, None])
                        frame.append({
                            "x": loc[0],
                            "y": loc[1],
                            "teammate": p.get("teammate", False),
                            "actor": p.get("actor", False),
                            "keeper": p.get("position", {}).get("name") == "Goalkeeper",
                        })
                    frames_out[ev_id] = frame

    with open(OUTPUT_DIR / "freeze_frames.json", "w") as f:
        json.dump(frames_out, f)
    print(f"  {len(frames_out)} freeze frames exported")

    # Export model_coefs.json (logistic regression coefficients for JS predict)
    print("Exporting model_coefs.json...")
    # Extract from pipeline
    lr_a = model_a.named_steps["model"]
    lr_b = model_b.named_steps["model"]
    prep_a = model_a.named_steps["prep"]
    prep_b = model_b.named_steps["prep"]

    # Get feature names after preprocessing
    feat_names_a = prep_a.get_feature_names_out()
    feat_names_b = prep_b.get_feature_names_out()

    coefs_out = {
        "model_a": {
            "features": list(feat_names_a),
            "coef": lr_a.coef_[0].tolist(),
            "intercept": float(lr_a.intercept_[0]),
        },
        "model_b": {
            "features": list(feat_names_b),
            "coef": lr_b.coef_[0].tolist(),
            "intercept": float(lr_b.intercept_[0]),
        },
    }

    with open(OUTPUT_DIR / "model_coefs.json", "w") as f:
        json.dump(coefs_out, f)
    print(f"  Model A: {len(feat_names_a)} features, Model B: {len(feat_names_b)} features")

    # Export evaluation_summary.json
    print("Exporting evaluation_summary.json...")
    cv_results = pd.read_csv(Path.home() / "data/outputs/model_training/cv_results_all_v2.csv")
    
    eval_out = {"kfold": [], "loto": [], "brier_calibration": []}
    
    for _, row in cv_results.iterrows():
        entry = {
            "model": f"{row['feature_set']}_{row['model']}",
            "roc_auc": round(row["roc_auc"], 4),
            "pr_auc": round(row["pr_auc"], 4),
            "f1": round(row["f1"], 4),
            "brier_score": round(row["brier_calibrated"], 4),
        }
        if row["validation"] == "stratified_kfold":
            eval_out["kfold"].append(entry)
        else:
            eval_out["loto"].append(entry)

    # Brier calibration comparison
    for _, row in cv_results.iterrows():
        if row["validation"] == "leave_one_tournament_out":
            eval_out["brier_calibration"].append({
                "model": f"{row['feature_set']}_{row['model']}",
                "brier_pre_kalibracije": round(row["brier_uncalibrated"], 4),
                "brier_posle_kalibracije": round(row["brier_calibrated"], 4),
            })

    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(eval_out, f)
    print("  Done")

    # Export shap_importance.json (from v2 SHAP if available, or compute)
    print("Exporting shap_importance.json...")
    try:
        import shap
        xgb_b = joblib.load(Path.home() / "data/models/model_b_360_v3_xgboost.pkl")
        X_b = df[feats_b].copy()
        # Transform through prep step
        X_b_transformed = xgb_b.named_steps["prep"].transform(X_b)
        feat_names_xgb = xgb_b.named_steps["prep"].get_feature_names_out()
        explainer = shap.TreeExplainer(xgb_b.named_steps["model"])
        shap_values = explainer.shap_values(X_b_transformed)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        shap_out = []
        for name, val in sorted(zip(feat_names_xgb, mean_abs_shap), key=lambda x: -x[1]):
            shap_out.append({"feature": str(name), "mean_abs_shap_B": round(float(val), 4)})
        
        with open(OUTPUT_DIR / "shap_importance.json", "w") as f:
            json.dump(shap_out, f)
        print(f"  {len(shap_out)} features exported")
    except Exception as e:
        print(f"  SHAP export failed: {e}, keeping old file")

    # Calibration maps not needed - v2 models have calibration built into Pipeline
    # Web app will use xg_a/xg_b directly from shots.json (already calibrated)
    print("Calibration: built into Pipeline, no separate maps needed.")
    print("  Writing empty calibration_maps.json (web app uses pre-computed xg_a/xg_b)")
    with open(OUTPUT_DIR / "calibration_maps.json", "w") as f:
        json.dump({"note": "v2 models have calibration built in, use xg_a/xg_b from shots.json directly"}, f)

    print(f"\nAll exports saved to: {OUTPUT_DIR}")
    print("DONE.")

if __name__ == "__main__":
    main()
