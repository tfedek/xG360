"""
08_paired_bootstrap_ci.py  -  Cluster bootstrap by match_id
================================================================
Paired cluster bootstrap 95% CI for the AUC difference (Model B - Model A)
computed on OUT-OF-FOLD (OOF) predictions pooled across all
Leave-One-Tournament-Out folds.

Method (cluster bootstrap):
  1. Collect OOF predictions for Model A and Model B logistic regression
     (same fold splits, so predictions are PAIRED per shot).
  2. Bootstrap: resample MATCHES (with replacement). For each selected match,
     include ALL its shots. Compute AUC_B - AUC_A on the bootstrap sample.
  3. Report 2.5th and 97.5th percentiles as 95% CI.

This respects the grouped structure: shots from the same match are not
independent, so we resample at the match level.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
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
    used = numeric_features + CATEGORICAL + [TARGET, "tournament", "match_id"]
    data = df[used].copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
        # NaN left for pipeline imputer
    for col in CATEGORICAL:
        data[col] = data[col].astype(str)
    data[TARGET] = data[TARGET].astype(int)
    return (data[numeric_features + CATEGORICAL], data[TARGET],
            data["tournament"], data["match_id"])


def collect_oof_predictions(df, numeric_features, model_name_key):
    X, y, groups_tournament, groups_match = prepare_xy(df, numeric_features)
    splitter = LeaveOneGroupOut()
    _, (estimator, grid) = list(make_models(numeric_features).items())[
        0 if model_name_key == "logistic" else 1]

    all_prob = np.zeros(len(y))

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups_tournament)):
        groups_match_train = groups_match.iloc[tr]
        inner_cv = StratifiedGroupKFold(n_splits=3)
        search = GridSearchCV(estimator, grid, scoring="average_precision",
                               cv=inner_cv, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups_match_train)
        all_prob[te] = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]

    return all_prob, y.values, groups_match.values


def main():
    df = load_modeling_data(DATASET_PATH)
    print("Collecting OOF predictions for Model A (logistic) ...")
    prob_a, y_true, match_ids = collect_oof_predictions(df, MODEL_A_NUMERIC, "logistic")
    print("Collecting OOF predictions for Model B (logistic) ...")
    prob_b, _, _ = collect_oof_predictions(df, MODEL_B_NUMERIC, "logistic")

    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)
    observed_diff = auc_b - auc_a
    print(f"\nOOF AUC  Model A: {auc_a:.4f}")
    print(f"OOF AUC  Model B: {auc_b:.4f}")
    print(f"Observed diff (B-A): {observed_diff:.4f}")

    # Build match-level index for cluster bootstrap
    unique_matches = np.unique(match_ids)
    match_to_idx = {m: np.where(match_ids == m)[0] for m in unique_matches}
    n_matches = len(unique_matches)
    print(f"\nCluster bootstrap: {n_matches} matches, {N_BOOT} iterations")

    # Cluster bootstrap: resample matches with replacement
    rng = np.random.default_rng(RANDOM_STATE)
    boot_diffs = []
    skipped = 0

    for _ in range(N_BOOT):
        # Sample matches with replacement
        sampled_matches = rng.choice(unique_matches, size=n_matches, replace=True)

        # Gather all shot indices for sampled matches
        idx = np.concatenate([match_to_idx[m] for m in sampled_matches])

        y_b = y_true[idx]
        # Skip if only one class present
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            skipped += 1
            continue

        auc_b_boot = roc_auc_score(y_b, prob_b[idx])
        auc_a_boot = roc_auc_score(y_b, prob_a[idx])
        boot_diffs.append(auc_b_boot - auc_a_boot)

    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_approx = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

    print(f"\n[P5] Cluster bootstrap CI (n_boot={len(boot_diffs)}/{N_BOOT}, skipped={skipped}):")
    print(f"  Observed diff: {observed_diff:.4f}")
    print(f"  Mean boot diff: {boot_diffs.mean():.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  CI contains 0: {'YES (not significant)' if ci_lo <= 0 <= ci_hi else 'NO (significant)'}")
    print(f"  Approx two-sided p: {p_approx:.4f}")

    # Verify: observed diff should be within CI (sanity check)
    if ci_lo <= observed_diff <= ci_hi:
        print("  Sanity: observed diff IS within CI. OK.")
    else:
        print("  WARNING: observed diff is OUTSIDE CI!")

    result = {
        "method": "cluster_bootstrap_by_match",
        "n_shots": len(y_true), "n_matches": n_matches,
        "oof_auc_a": auc_a, "oof_auc_b": auc_b,
        "observed_diff": observed_diff,
        "ci_lower_95": ci_lo, "ci_upper_95": ci_hi,
        "p_approx": p_approx,
        "n_boot": len(boot_diffs), "n_skipped": skipped,
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / "paired_bootstrap_ci_oof_v2.csv", index=False)
    print(f"Saved: {OUT_DIR / 'paired_bootstrap_ci_oof_v2.csv'}")


if __name__ == "__main__":
    main()
