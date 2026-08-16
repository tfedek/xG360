from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from football_xg.config import (
    DATASET_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
    MODEL_A_NUMERIC,
    MODEL_B_NUMERIC,
    CATEGORICAL,
    TARGET,
)
from football_xg.modeling import load_modeling_data

OUT_DIR = OUTPUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "A Logistic": ("model_a_classic_logistic.pkl", MODEL_A_NUMERIC),
    "A XGBoost": ("model_a_classic_xgboost.pkl", MODEL_A_NUMERIC),
    "B Logistic": ("model_b_360_v3_logistic.pkl", MODEL_B_NUMERIC),
    "B XGBoost": ("model_b_360_v3_xgboost.pkl", MODEL_B_NUMERIC),
}


def prepare_x(df, features):
    """
    Cleaning used only for model prediction.
    These values are NOT used for display in the app.
    """
    data = df.copy()

    for col in features:
        if col not in data.columns:
            data[col] = np.nan

        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        median_value = data[col].median()

        if pd.isna(median_value):
            median_value = 0

        data[col] = data[col].fillna(median_value)

    for col in CATEGORICAL:
        if col not in data.columns:
            data[col] = "Unknown"

        data[col] = data[col].fillna("Unknown").astype(str)

    X = data[features + CATEGORICAL]

    if X.isna().sum().sum() > 0:
        print("\nNaN columns still present:")
        print(X.isna().sum()[X.isna().sum() > 0])
        raise ValueError("X still contains NaN after prepare_x().")

    return X


def main():
    # RAW dataset for display/export
    raw_df = pd.read_csv(DATASET_PATH)

    # CLEAN dataset for model prediction only
    model_df = load_modeling_data(DATASET_PATH)

    y = raw_df[TARGET].astype(int)

    base_cols = [
        "id",
        "match_id",
        "tournament",
        "match_date",
        "team",
        "player",
        "minute",
        "second",
        "x",
        "y",
        "goal",
        "shot_outcome",
        "distance",
        "angle",
        "shot_body_part",
        "shot_type",
        "shot_technique",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",

        # 360 raw display features
        "goalkeeper_distance_360",
        "goalkeeper_distance_to_goal_360",
        "goalkeeper_in_shot_cone_360",
        "goalkeeper_offset_from_goal_center_360",

        "nearest_defender_distance_360",
        "second_nearest_defender_distance_360",
        "mean_defender_distance_360",
        "std_defender_distance_360",

        "defenders_within_2m_360",
        "defenders_within_5m_360",
        "defenders_within_10m_360",

        "defenders_in_shot_cone_360",
        "opponents_between_shooter_and_goal_360",

        "nearest_defender_to_shot_line_360",
        "defenders_within_1m_of_shot_line_360",
        "defenders_within_2m_of_shot_line_360",

        "blocked_angle_360",
        "open_angle_ratio_360",
        "visible_goal_ratio_360",

        "free_space_radius_360",
        "pressure_score_360",
        "local_opponent_density_5m_360",
        "local_opponent_density_10m_360",

        "opponents_in_box_360",
        "teammates_in_box_360",
        "teammates_ahead_of_ball_360",
    ]

    predictions = pd.DataFrame()

    # Use RAW values for display
    for col in base_cols:
        if col in raw_df.columns:
            predictions[col] = raw_df[col]
        else:
            predictions[col] = np.nan

    predictions["goal"] = y

    # ROC
    plt.figure(figsize=(7, 6))

    for label, (file_name, features) in MODELS.items():
        path = MODELS_DIR / file_name

        if not path.exists():
            print(f"Skipping missing model: {path}")
            continue

        pipe = joblib.load(path)

        # Use CLEANED values only for model prediction
        X = prepare_x(model_df, features)
        prob = pipe.predict_proba(X)[:, 1]

        col_name = f"xg_{label.replace(' ', '_').lower()}"
        predictions[col_name] = prob

        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{label} AUC={auc(fpr, tpr):.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=180)
    plt.close()

    # PR
    plt.figure(figsize=(7, 6))

    for label, (file_name, features) in MODELS.items():
        path = MODELS_DIR / file_name

        if not path.exists():
            continue

        pipe = joblib.load(path)

        X = prepare_x(model_df, features)
        prob = pipe.predict_proba(X)[:, 1]

        precision, recall, _ = precision_recall_curve(y, prob)
        plt.plot(recall, precision, label=f"{label} PR-AUC={auc(recall, precision):.3f}")

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curves.png", dpi=180)
    plt.close()

    # Calibration
    plt.figure(figsize=(7, 6))

    for label, (file_name, features) in MODELS.items():
        path = MODELS_DIR / file_name

        if not path.exists():
            continue

        pipe = joblib.load(path)

        X = prepare_x(model_df, features)
        prob = pipe.predict_proba(X)[:, 1]

        prob_true, prob_pred = calibration_curve(
            y,
            prob,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Calibration Curves")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calibration_curves.png", dpi=180)
    plt.close()

    if "xg_a_logistic" in predictions.columns and "xg_b_logistic" in predictions.columns:
        predictions["xg_diff_b_minus_a_logistic"] = (
            predictions["xg_b_logistic"] - predictions["xg_a_logistic"]
        )

    pred_path = OUT_DIR / "all_shot_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    print(f"Saved figures and predictions in: {OUT_DIR}")
    print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()