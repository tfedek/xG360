from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from football_xg.config import DATASET_PATH, OUTPUT_DIR, MODELS_DIR, TARGET, MODEL_B_NUMERIC, CATEGORICAL
from football_xg.modeling import load_modeling_data

OUT_DIR = OUTPUT_DIR / "interpretation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def logistic_odds_ratios():
    df = load_modeling_data(DATASET_PATH)
    X_num = df[MODEL_B_NUMERIC].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    X = sm.add_constant(X)
    y = df[TARGET].astype(int)

    model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    params = model.params
    ci = model.conf_int()
    out = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_low": np.exp(ci[0].values),
        "ci_high": np.exp(ci[1].values),
        "p_value": model.pvalues.values,
    }).sort_values("p_value")

    out.to_csv(OUT_DIR / "logistic_odds_ratios_model_b.csv", index=False)
    print("\\n=== TOP LOGISTIC ODDS RATIOS / SIGNIFICANCE ===")
    print(out.head(30).to_string(index=False))

def model_feature_importance():
    model_path = MODELS_DIR / "model_b_360_v3_xgboost.pkl"
    if not model_path.exists():
        print(f"Missing {model_path}; skipping XGBoost feature importance.")
        return

    df = load_modeling_data(DATASET_PATH)
    pipe = joblib.load(model_path)
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_

    out = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    out.to_csv(OUT_DIR / "xgboost_feature_importance_model_b.csv", index=False)

    plt.figure(figsize=(9, 7))
    top = out.head(25).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("XGBoost Feature Importance - Model B")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xgboost_feature_importance_model_b.png", dpi=180)
    plt.close()

    print("\\n=== TOP XGBOOST IMPORTANCE ===")
    print(out.head(25).to_string(index=False))

def main():
    logistic_odds_ratios()
    model_feature_importance()
    print(f"\\nSaved interpretation outputs: {OUT_DIR}")

if __name__ == "__main__":
    main()
