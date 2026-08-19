"""
03_train_models.py  –  revision v2 (professor review)
========================================================
Changes vs. v1:
  [P1] Isotonic calibration now fitted INSIDE each CV fold on the
       training split → calibrated Brier reported on the held-out fold
       (no calibration leakage from test data).
  [P2] class_weight is part of the hyperparameter grid, so its
       interaction with calibration is handled per fold.
  [P3] goalkeeper_anomaly residuals would be fitted here if used;
       in current feature set this is handled upstream in
       02_build_dataset.py → no leakage from global fit.
  [P6] A separate UNPENALIZED, UNWEIGHTED logistic regression
       (statsmodels Logit, full dataset) is used for the formal
       Likelihood Ratio test, AIC and BIC. This is methodologically
       correct: regularised / class-weighted sklearn models break
       the classical inference assumptions.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from football_xg.config import (
    DATASET_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
    TARGET,
    MODEL_A_NUMERIC,
    MODEL_B_NUMERIC,
    CATEGORICAL,
    RANDOM_STATE,
)
from football_xg.data_utils import ensure_dirs
from football_xg.modeling import load_modeling_data, make_models, metrics

ensure_dirs(MODELS_DIR, OUTPUT_DIR / "model_training")


# ============================================================
# Helper: prepare X, y, groups
# ============================================================
def prepare_xy(df, numeric_features):
    used_cols = numeric_features + CATEGORICAL + [TARGET, "tournament", "match_id"]
    data = df[used_cols].copy()

    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        median_value = data[col].median()
        if pd.isna(median_value):
            median_value = 0
        data[col] = data[col].fillna(median_value)

    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str)

    data[TARGET] = data[TARGET].astype(int)
    data["tournament"] = data["tournament"].astype(str)
    data["match_id"] = data["match_id"].astype(str)

    X = data[numeric_features + CATEGORICAL]
    y = data[TARGET]
    groups_tournament = data["tournament"]
    groups_match = data["match_id"]

    if X.isna().sum().sum() > 0:
        raise ValueError("X still contains NaN after prepare_xy().")

    return X, y, groups_tournament, groups_match


# ============================================================
# [P1] CV with in-fold isotonic calibration
# ============================================================
def run_cv_with_calibration(df, feature_set, numeric_features, model_name,
                             estimator, grid, validation):
    """
    For each outer fold:
      1. Inner GridSearchCV selects best hyperparameters on training split.
      2. Best estimator is re-wrapped with CalibratedClassifierCV (isotonic,
         cv=3) fitted ONLY on the training split.
      3. Calibrated model predicts on the held-out fold.

    This avoids any information from the test split entering the calibrator.
    """
    X, y, groups_tournament, groups_match = prepare_xy(df, numeric_features)

    if validation == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(n_splits=5)
        split_iter = splitter.split(X, y, groups_match)
    elif validation == "leave_one_tournament_out":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, y, groups_tournament)
    else:
        raise ValueError(f"Unknown validation: {validation}")

    rows = []
    all_true, all_prob, all_prob_uncal = [], [], []

    for fold, (tr, te) in enumerate(split_iter, 1):
        print(f"  {feature_set} | {model_name} | {validation} | fold {fold}")

        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        groups_match_train = groups_match.iloc[tr]

        inner_cv = StratifiedGroupKFold(n_splits=3)

        search = GridSearchCV(
            estimator, grid,
            scoring="average_precision",
            cv=inner_cv, n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups_match_train)
        best_est = search.best_estimator_

        # [P1] Calibrate INSIDE the fold (only on training data)
        calibrated = CalibratedClassifierCV(
            estimator=best_est, method="isotonic", cv=3
        )
        calibrated.fit(X_train, y_train)

        prob_cal = calibrated.predict_proba(X_test)[:, 1]
        prob_raw = best_est.predict_proba(X_test)[:, 1]

        m = metrics(y_test, prob_cal)
        m["brier_uncalibrated"] = brier_score_loss(y_test, prob_raw)
        m["brier_calibrated"]   = brier_score_loss(y_test, prob_cal)
        m.update({
            "feature_set": feature_set,
            "model": model_name,
            "validation": validation,
            "fold": fold,
            "best_params": str(search.best_params_),
            "test_tournament": groups.iloc[te].iloc[0]
                               if validation == "leave_one_tournament_out" else "",
        })

        rows.append(m)
        all_true.extend(y_test.tolist())
        all_prob.extend(prob_cal.tolist())
        all_prob_uncal.extend(prob_raw.tolist())

    overall = metrics(np.array(all_true), np.array(all_prob))
    overall["brier_uncalibrated"] = brier_score_loss(np.array(all_true), np.array(all_prob_uncal))
    overall["brier_calibrated"]   = brier_score_loss(np.array(all_true), np.array(all_prob))
    overall.update({
        "feature_set": feature_set, "model": model_name,
        "validation": validation, "fold": "overall",
        "best_params": "", "test_tournament": "",
    })
    rows.append(overall)
    return pd.DataFrame(rows)


# ============================================================
# [P6] Formal LR test: UNPENALIZED, UNWEIGHTED statsmodels Logit
#      Used ONLY for AIC/BIC/LR-test — NOT for Brier/AUC reporting
# ============================================================
def run_lr_test_statsmodels(df):
    """
    Fits two unpenalized logistic regressions (statsmodels Logit) on
    the full dataset:
      Model A: classic features only
      Model B: classic + 360 features
    Then runs the Likelihood Ratio test (B nested in A).

    [P6] Using statsmodels without regularisation / class_weight so that
    AIC, BIC, and the chi-square p-value are statistically valid.
    """
    print("\n[P6] Formal LR test (unpenalized statsmodels Logit) ...")

    # [P6+FIX] Isključi penale iz LR testa — shot_type_Penalty uzrokuje
    # kvazi-savršenu separaciju u Model B i dovodi do llf=-inf.
    # Ovo je konzistentno sa sklearn CV koji ih isključuje kroz CATEGORICAL
    # one-hot encoding (ali tamo class_weight/regularizacija sprečava divergenciju).
    if "shot_type" in df.columns:
        df_no_pen = df[df["shot_type"] != "Penalty"].copy()
        print(f"  Penali isključeni: {len(df) - len(df_no_pen)} šuteva uklonjeno. n={len(df_no_pen)}")
    else:
        df_no_pen = df.copy()

    results = {}
    for label, numeric in [("A_classic", MODEL_A_NUMERIC),
                             ("B_360",     MODEL_B_NUMERIC)]:
        X_num, y, _, _ = prepare_xy(df_no_pen, numeric)
        # One-hot encode categoricals manually (statsmodels needs numpy)
        X_cat = pd.get_dummies(df_no_pen[CATEGORICAL], drop_first=True)
        X_all = pd.concat([X_num[numeric], X_cat], axis=1).astype(float)
        X_all = sm.add_constant(X_all)

        # Drop NaN rows (identical subset to sklearn pipeline)
        mask = X_all.notna().all(axis=1)
        X_fit = X_all[mask]
        y_fit = y[mask]

        fit = sm.Logit(y_fit, X_fit).fit(disp=False, maxiter=300)
        results[label] = fit
        print(f"  Model {label}: llf={fit.llf:.4f}, AIC={fit.aic:.2f}, BIC={fit.bic:.2f}, n={len(y_fit)}")

    fit_a = results["A_classic"]
    fit_b = results["B_360"]
    df_diff = fit_b.df_model - fit_a.df_model
    lr_stat = 2 * (fit_b.llf - fit_a.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)

    print(f"\n  LR statistic = {lr_stat:.4f}, df = {df_diff:.0f}, p = {p_value:.4e}")
    print(f"  AIC A={fit_a.aic:.2f} | B={fit_b.aic:.2f}")
    print(f"  BIC A={fit_a.bic:.2f} | B={fit_b.bic:.2f}")

    return {
        "llf_a": fit_a.llf, "llf_b": fit_b.llf,
        "aic_a": fit_a.aic, "aic_b": fit_b.aic,
        "bic_a": fit_a.bic, "bic_b": fit_b.bic,
        "lr_statistic": lr_stat, "df_diff": df_diff, "p_value": p_value,
    }


# ============================================================
# Fit final model (for export / web app)
# ============================================================
def fit_final(df, feature_set, numeric_features, model_name, estimator, grid):
    X, y, _, groups_match = prepare_xy(df, numeric_features)
    cv = StratifiedGroupKFold(n_splits=5)
    search = GridSearchCV(
        estimator, grid, scoring="average_precision",
        cv=cv, n_jobs=-1, refit=True, error_score="raise",
    )
    search.fit(X, y, groups=groups_match)
    path = MODELS_DIR / f"{feature_set}_{model_name}.pkl"
    joblib.dump(search.best_estimator_, path)
    return path, search.best_params_, search.best_score_


# ============================================================
# main
# ============================================================
def main():
    df = load_modeling_data(DATASET_PATH)
    print(f"Rows (ukupno, sa penalima): {len(df)} | Goals: {df[TARGET].sum()} ({df[TARGET].mean():.3f})")

    # Isključi penale iz svih modela — konzistentno sa originalnim pipeline-om.
    # Penali nemaju 360 podatke (osim blokiranih), fiksne su geometrije, i
    # uvode kvazi-savršenu separaciju u nepenalizovanom LR testu.
    if "shot_type" in df.columns:
        df = df[df["shot_type"] != "Penalty"].copy().reset_index(drop=True)
        print(f"Rows (bez penala): {len(df)} | Goals: {df[TARGET].sum()} ({df[TARGET].mean():.3f})")

    feature_sets = {
        "model_a_classic": MODEL_A_NUMERIC,
        "model_b_360_v3":  MODEL_B_NUMERIC,
    }

    all_results = []
    final_rows = []

    for feature_set, features in feature_sets.items():
        for model_name, (estimator, grid) in make_models(features).items():
            for validation in ["stratified_group_kfold", "leave_one_tournament_out"]:
                result = run_cv_with_calibration(
                    df=df, feature_set=feature_set,
                    numeric_features=features, model_name=model_name,
                    estimator=estimator, grid=grid, validation=validation,
                )
                all_results.append(result)

            path, params, score = fit_final(
                df=df, feature_set=feature_set,
                numeric_features=features, model_name=model_name,
                estimator=estimator, grid=grid,
            )
            final_rows.append({
                "feature_set": feature_set, "model": model_name,
                "path": str(path), "best_params": str(params),
                "best_cv_pr_auc": score,
            })

    out_dir = OUTPUT_DIR / "model_training"
    ensure_dirs(out_dir)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(out_dir / "cv_results_all_v2.csv", index=False)
    pd.DataFrame(final_rows).to_csv(out_dir / "final_models.csv", index=False)

    # [P6] Formal LR test (unpenalized)
    lr_results = run_lr_test_statsmodels(df)
    pd.DataFrame([lr_results]).to_csv(out_dir / "lr_test_results_v2.csv", index=False)

    print("\n=== OVERALL RESULTS (calibrated Brier) ===")
    overall = results[results["fold"].astype(str) == "overall"]
    print(overall[[
        "feature_set", "model", "validation",
        "roc_auc", "pr_auc", "f1",
        "brier_uncalibrated", "brier_calibrated",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()