"""
05_model_interpretation_v2.py  –  revision v2 (professor review)
==================================================================
Changes vs. v1:
  [P7] Cluster-robust standard errors (clustered by match_id) added to
       the unpenalized statsmodels Logit, as a sensitivity check.
       This accounts for the fact that shots within the same match are
       not independent (shared game context, fatigue, tactics).
  [P9] Language in printed output uses "association" / "predictive
       contribution" rather than "causal" or strong causal language.
       The final paragraph in the Word document should mirror this.
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, MODELS_DIR, TARGET,
    MODEL_B_NUMERIC, CATEGORICAL,
)
from football_xg.modeling import load_modeling_data
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "interpretation"
ensure_dirs(OUT_DIR)


def logistic_odds_ratios_with_cluster_se():
    """
    [P7] Fits unpenalized statsmodels Logit on Model B features and
    computes BOTH:
      (a) standard sandwich (HC3) standard errors
      (b) cluster-robust standard errors (clustered by match_id)
    as a sensitivity check for the within-match non-independence.

    [P9] Outputs are labelled as "predictive association", not causal.
    """
    df = load_modeling_data(DATASET_PATH)

    # Need match_id for clustering — load from raw CSV
    raw = pd.read_csv(DATASET_PATH)
    if "match_id" not in raw.columns:
        print("  [P7] match_id column not found; skipping cluster-robust SE.")
        cluster_groups = None
    else:
        cluster_groups = raw["match_id"].astype(str)

    X_num = df[MODEL_B_NUMERIC].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    X = sm.add_constant(X)
    y = df[TARGET].astype(int)

    # Standard fit
    model = sm.Logit(y, X).fit(disp=False, maxiter=200)

    params = model.params
    ci_std = model.conf_int()

    out_std = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_low_std": np.exp(ci_std[0].values),
        "ci_high_std": np.exp(ci_std[1].values),
        "p_value_std": model.pvalues.values,
    })

    # [P7] Cluster-robust SEs
    if cluster_groups is not None:
        robust_fit = sm.Logit(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_groups.values},
            disp=False, maxiter=200,
        )
        ci_rob = robust_fit.conf_int()
        out_std["p_value_cluster_robust"] = robust_fit.pvalues.values
        out_std["ci_low_cluster_robust"] = np.exp(ci_rob.iloc[:, 0])
        out_std["ci_high_cluster_robust"] = np.exp(ci_rob.iloc[:, 1])

        print("\n[P7] Cluster-robust vs. standard p-values (top features by standard p):")
        compare = out_std[["feature", "p_value_std", "p_value_cluster_robust"]].sort_values("p_value_std").head(15)
        print(compare.to_string(index=False))

    out_std = out_std.sort_values("p_value_std")
    out_std.to_csv(OUT_DIR / "logistic_odds_ratios_model_b_v2.csv", index=False)

    # [P9] Print with careful language
    print("\n[P9] Predictive associations (Model B, unpenalized logistic, odds ratios):")
    print("     Note: these reflect statistical association in the fitted model,")
    print("     not causal effects.")
    print(out_std[["feature", "odds_ratio", "p_value_std"]].head(20).to_string(index=False))


def model_feature_importance():
    model_path = MODELS_DIR / "model_b_360_v3_xgboost.pkl"
    if not model_path.exists():
        print(f"Missing {model_path}; skipping XGBoost feature importance.")
        return

    df = load_modeling_data(DATASET_PATH)
    pipe = joblib.load(model_path)
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_

    out = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    out.to_csv(OUT_DIR / "xgboost_feature_importance_model_b_v2.csv", index=False)

    plt.figure(figsize=(9, 7))
    top = out.head(25).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("XGBoost Feature Importance – Model B\n(predictive contribution, not causal)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xgboost_feature_importance_model_b_v2.png", dpi=180)
    plt.close()

    print("\n[P9] XGBoost feature importances (predictive contribution):")
    print(out.head(20).to_string(index=False))


def main():
    logistic_odds_ratios_with_cluster_se()
    model_feature_importance()
    print(f"\nSaved interpretation outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
