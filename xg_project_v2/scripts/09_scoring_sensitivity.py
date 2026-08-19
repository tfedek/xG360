"""
09_scoring_sensitivity.py  -  Does the choice of scoring metric for GridSearch
                              change the conclusion that Model B > Model A?
================================================================
Tests: average_precision (default), neg_log_loss, neg_brier_score
For each scoring metric, re-runs LOTO CV with GridSearchCV and reports
AUC and Brier for both models.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL, RANDOM_STATE,
)
from football_xg.modeling import load_modeling_data, make_models
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "scoring_sensitivity"
ensure_dirs(OUT_DIR)

SCORING_METRICS = ["average_precision", "neg_log_loss", "neg_brier_score"]


def prepare_xy(df, numeric_features):
    used = numeric_features + CATEGORICAL + [TARGET, "tournament", "match_id"]
    data = df[used].copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    for col in CATEGORICAL:
        data[col] = data[col].astype(str)
    data[TARGET] = data[TARGET].astype(int)
    return (data[numeric_features + CATEGORICAL], data[TARGET],
            data["tournament"], data["match_id"])


def loto_eval(df, numeric_features, scoring):
    X, y, groups_tournament, groups_match = prepare_xy(df, numeric_features)
    splitter = LeaveOneGroupOut()
    _, (estimator, grid) = list(make_models(numeric_features).items())[0]  # logistic

    all_prob = np.zeros(len(y))

    for tr, te in splitter.split(X, y, groups_tournament):
        groups_match_train = groups_match.iloc[tr]
        inner_cv = StratifiedGroupKFold(n_splits=3)
        search = GridSearchCV(estimator, grid, scoring=scoring,
                               cv=inner_cv, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups_match_train)
        all_prob[te] = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]

    auc = roc_auc_score(y, all_prob)
    brier = brier_score_loss(y, all_prob)
    return auc, brier


def main():
    df = load_modeling_data(DATASET_PATH)
    print("=== Scoring Sensitivity Analysis ===\n")

    rows = []
    for scoring in SCORING_METRICS:
        print(f"Scoring: {scoring}")
        auc_a, brier_a = loto_eval(df, MODEL_A_NUMERIC, scoring)
        auc_b, brier_b = loto_eval(df, MODEL_B_NUMERIC, scoring)
        delta_auc = auc_b - auc_a
        delta_brier = brier_b - brier_a

        row = {
            "scoring": scoring,
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta_auc": round(delta_auc, 4),
            "brier_a": round(brier_a, 4), "brier_b": round(brier_b, 4),
            "delta_brier": round(delta_brier, 4),
        }
        rows.append(row)
        print(f"  Model A: AUC={auc_a:.4f}, Brier={brier_a:.4f}")
        print(f"  Model B: AUC={auc_b:.4f}, Brier={brier_b:.4f}")
        print(f"  Delta AUC: {delta_auc:+.4f}, Delta Brier: {delta_brier:+.4f}")
        print()

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "scoring_sensitivity.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'scoring_sensitivity.csv'}")

    # Conclusion
    all_positive = all(r["delta_auc"] > 0 for r in rows)
    print(f"\nModel B > Model A across ALL scoring metrics: {'YES' if all_positive else 'NO'}")


if __name__ == "__main__":
    main()
