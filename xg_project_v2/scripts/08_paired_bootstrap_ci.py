"""
08_paired_bootstrap_ci.py  –  NEW (professor review, point P5)
================================================================
Paired bootstrap 95% CI for the AUC difference (Model B - Model A)
computed on OUT-OF-FOLD (OOF) predictions pooled across all
Leave-One-Tournament-Out folds.

Why OOF predictions are better than a single 80/20 split:
  - Uses all available data for evaluation
  - Each prediction is genuinely out-of-sample
  - Bootstrap on the full OOF set is less sensitive to split randomness

Method:
  1. Collect OOF predictions for Model A and Model B logistic regression
     (same fold splits, so predictions are PAIRED per shot).
  2. Bootstrap: resample (with replacement) the paired (y, p_A, p_B)
     rows 2000 times, compute AUC_B - AUC_A each time.
  3. Report 2.5th and 97.5th percentiles as 95% CI.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL, RANDOM_STATE,
)
from football_xg.modeling import load_modeling_data, make_models
from football_xg.data_utils import ensure_dirs

N_BOOT = 2000
OUT_DIR = OUTPUT_DIR / "model_training"
ensure_dirs(OUT_DIR)


def prepare_xy(df, numeric_features):
    used = numeric_features + CATEGORICAL + [TARGET, "tournament"]
    data = df[used].copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        data[col] = data[col].fillna(data[col].median())
    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str)
    data[TARGET] = data[TARGET].astype(int)
    return data[numeric_features + CATEGORICAL], data[TARGET], data["tournament"]


def collect_oof_predictions(df, numeric_features, model_name_key):
    X, y, groups = prepare_xy(df, numeric_features)
    splitter = LeaveOneGroupOut()
    _, (estimator, grid) = list(make_models(numeric_features).items())[
        0 if model_name_key == "logistic" else 1]

    all_prob = np.zeros(len(y))
    all_idx  = np.zeros(len(y), dtype=int)

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(estimator, grid, scoring="average_precision",
                               cv=inner_cv, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr])
        all_prob[te] = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        all_idx[te] = te

    return all_prob, y.values


def main():
    df = load_modeling_data(DATASET_PATH)
    print("Collecting OOF predictions for Model A (logistic) ...")
    prob_a, y_true = collect_oof_predictions(df, MODEL_A_NUMERIC, "logistic")
    print("Collecting OOF predictions for Model B (logistic) ...")
    prob_b, _      = collect_oof_predictions(df, MODEL_B_NUMERIC, "logistic")

    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)
    observed_diff = auc_b - auc_a
    print(f"\nOOF AUC  Model A: {auc_a:.4f}")
    print(f"OOF AUC  Model B: {auc_b:.4f}")
    print(f"Observed diff (B-A): {observed_diff:.4f}")

    # [P5] Paired bootstrap on OOF predictions
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boot_diffs = []

    for _ in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        d = roc_auc_score(y_b, prob_b[idx]) - roc_auc_score(y_b, prob_a[idx])
        boot_diffs.append(d)

    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_approx = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

    print(f"\n[P5] Paired bootstrap CI (n_boot={len(boot_diffs)}/{N_BOOT}):")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  CI contains 0: {'YES (not significant)' if ci_lo <= 0 <= ci_hi else 'NO (significant)'}")
    print(f"  Approx two-sided p: {p_approx:.4f}")

    result = {
        "n_shots": n, "oof_auc_a": auc_a, "oof_auc_b": auc_b,
        "observed_diff": observed_diff,
        "ci_lower_95": ci_lo, "ci_upper_95": ci_hi,
        "p_approx": p_approx, "n_boot": len(boot_diffs),
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / "paired_bootstrap_ci_oof_v2.csv", index=False)
    print(f"Saved: {OUT_DIR / 'paired_bootstrap_ci_oof_v2.csv'}")


if __name__ == "__main__":
    main()
