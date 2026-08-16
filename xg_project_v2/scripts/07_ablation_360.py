"""
07_ablation_360.py  –  NEW (professor review, point P4)
=========================================================
Ablation study: which 360 features contribute most to Model B's
advantage over Model A?

For each 360 feature group, trains Model A + that group and compares
ROC-AUC (LOTO) to:
  - Model A baseline (no 360)
  - Model B full (all 360)

This isolates the marginal contribution of each spatial feature group.

Groups (based on config.py MODEL_B_NUMERIC):
  GK         : goalkeeper_distance_360
  PRESSURE   : nearest_defender_distance_360, defenders_within_5m_360,
               defenders_within_10m_360, pressure_score_360
  CONE       : defenders_in_shot_cone_360, opponents_between_shooter_and_goal_360,
               opponents_in_box_360
  SHOT_LINE  : nearest_defender_to_shot_line_360,
               defenders_within_1m_of_shot_line_360,
               defenders_within_2m_of_shot_line_360
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL, RANDOM_STATE,
)
from football_xg.modeling import load_modeling_data
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "ablation"
ensure_dirs(OUT_DIR)

# 360-only feature groups
GROUPS_360 = {
    "GK":        ["goalkeeper_distance_360"],
    "PRESSURE":  ["nearest_defender_distance_360", "defenders_within_5m_360",
                   "defenders_within_10m_360", "pressure_score_360"],
    "CONE":      ["defenders_in_shot_cone_360",
                   "opponents_between_shooter_and_goal_360",
                   "opponents_in_box_360"],
    "SHOT_LINE": ["nearest_defender_to_shot_line_360",
                   "defenders_within_1m_of_shot_line_360",
                   "defenders_within_2m_of_shot_line_360"],
    "OPEN_ANGLE": ["open_angle_ratio_360"],
}


def make_pipe(numeric_features):
    prep = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = LogisticRegression(C=0.5, max_iter=4000, random_state=RANDOM_STATE)
    return Pipeline([("prep", prep), ("model", model)])


def loto_auc(df, numeric_features):
    """Leave-One-Tournament-Out average ROC-AUC."""
    used = numeric_features + CATEGORICAL + [TARGET, "tournament"]
    data = df[used].copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())
    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str)
    data[TARGET] = data[TARGET].astype(int)

    X = data[numeric_features + CATEGORICAL]
    y = data[TARGET]
    groups = data["tournament"]

    splitter = LeaveOneGroupOut()
    aucs = []
    for tr, te in splitter.split(X, y, groups):
        pipe = make_pipe(numeric_features)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
    return float(np.mean(aucs))


def main():
    df = load_modeling_data(DATASET_PATH)

    rows = []

    # Baseline: Model A (no 360)
    auc_a = loto_auc(df, MODEL_A_NUMERIC)
    rows.append({"variant": "Model A (baseline, no 360)", "auc_loto": auc_a,
                  "delta_vs_a": 0.0})
    print(f"Model A baseline LOTO AUC: {auc_a:.4f}")

    # Full Model B
    auc_b = loto_auc(df, MODEL_B_NUMERIC)
    rows.append({"variant": "Model B (full 360)", "auc_loto": auc_b,
                  "delta_vs_a": auc_b - auc_a})
    print(f"Model B full LOTO AUC:     {auc_b:.4f}  (delta={auc_b-auc_a:+.4f})")

    # Ablation: Model A + one group at a time
    for group_name, extra_features in GROUPS_360.items():
        features = MODEL_A_NUMERIC + extra_features
        auc = loto_auc(df, features)
        delta = auc - auc_a
        rows.append({"variant": f"Model A + {group_name}", "auc_loto": auc,
                      "delta_vs_a": delta})
        print(f"Model A + {group_name:<12}: LOTO AUC={auc:.4f}  (delta={delta:+.4f})")

    result = pd.DataFrame(rows).sort_values("auc_loto", ascending=False)
    result.to_csv(OUT_DIR / "ablation_360_loto_auc.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'ablation_360_loto_auc.csv'}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
