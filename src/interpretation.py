"""
interpretation.py
====================
Korak 8 (interpretacija) metodologije: Odds Ratio za logističku regresiju,
SHAP za XGBoost, i poređenje važnosti atributa Model A vs Model B.

Generiše:
  - tables/odds_ratios_model_a.csv, odds_ratios_model_b.csv
  - tables/shap_importance_model_a.csv, shap_importance_model_b.csv
  - figures/shap_summary_model_a.png, shap_summary_model_b.png
  - figures/shap_dependence_*.png (distance, angle, n_defenders)

Pokretanje: python3 interpretation.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # bez GUI-ja, samo eksport u fajl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
MODELS_DIR = PROCESSED_DIR / "fitted_models"

META_COLS = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]


def _load_xy(csv_name: str):
    df = pd.read_csv(PROCESSED_DIR / csv_name)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].astype(float)
    y = df["is_goal"].astype(int)
    return X, y


def odds_ratios_table(model_name: str, csv_name: str) -> pd.DataFrame:
    """
    Izvlači koeficijente logističke regresije, transformiše u Odds Ratio
    (exp(beta)), uz Wald 95% interval poverenja. Koristi statsmodels Logit
    (čist MLE) - problem kvazi-savršene separacije iz ranije verzije ovog
    koda je rešen NA IZVORU PODATAKA (vidi preprocessing.py / build_model_frames:
    retke shot_type kategorije Corner/Free Kick su agregovane u "Set Piece"
    pre kodiranja), ne post-hoc regularizacijom - čistije i transparentnije
    za izveštavanje p-vrednosti i intervala poverenja u radu.
    """
    import statsmodels.api as sm

    X, y = _load_xy(csv_name)
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0, maxiter=200)

    summary = pd.DataFrame({
        "coef": model.params, "std_err": model.bse, "p_value": model.pvalues,
    })
    summary["odds_ratio"] = np.exp(summary["coef"])
    summary["ci_lower"] = np.exp(summary["coef"] - 1.96 * summary["std_err"])
    summary["ci_upper"] = np.exp(summary["coef"] + 1.96 * summary["std_err"])
    summary = summary.drop(index="const").sort_values("odds_ratio", ascending=False)

    out_path = TABLES_DIR / f"odds_ratios_{model_name}.csv"
    summary.round(4).to_csv(out_path)
    print(f"\n-- Odds Ratio: {model_name} --")
    print(summary[["odds_ratio", "ci_lower", "ci_upper", "p_value"]].round(3).to_string())
    return summary


def shap_analysis(model_name: str, csv_name: str, top_n_dependence: int = 3):
    """
    SHAP TreeExplainer na fitovanom XGBoost modelu. Pravi:
      - summary plot (beeswarm) -> globalna važnost + smer uticaja
      - dependence plot za top_n_dependence najuticajnijih NUMERIČKIH
        atributa (po srednjoj |SHAP| vrednosti) - bira se IZ REZULTATA,
        ne unapred imenovano, po dogovoru iz ranije faze metodologije.
    """
    bundle = joblib.load(MODELS_DIR / f"{model_name}.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    X, y = _load_xy(csv_name)
    X = X[feature_names]  # osiguravamo isti redosled kolona kao pri treningu

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Globalna važnost (srednja apsolutna SHAP vrednost po feature-u)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(TABLES_DIR / f"shap_importance_{model_name}.csv", index=False)
    print(f"\n-- SHAP važnost (top 10): {model_name} --")
    print(importance.head(10).to_string(index=False))

    # Summary (beeswarm) plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_summary_{model_name}.png", dpi=150)
    plt.close()

    # Dependence plots za top-N numeričke (kontinualne, ne dummy 0/1) atribute
    numeric_candidates = [
        f for f in importance["feature"]
        if X[f].nunique() > 2  # izbacujemo binarne/dummy kolone iz dependence plot izbora
    ]
    top_numeric = numeric_candidates[:top_n_dependence]
    for feat in top_numeric:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat, shap_values, X, show=False)
        plt.tight_layout()
        safe_name = feat.replace(" ", "_").replace("/", "_")
        plt.savefig(FIGURES_DIR / f"shap_dependence_{model_name}_{safe_name}.png", dpi=150)
        plt.close()

    print(f"  Dependence plot-ovi generisani za: {top_numeric}")
    return importance


def compare_importance_a_vs_b(imp_a: pd.DataFrame, imp_b: pd.DataFrame) -> pd.DataFrame:
    """
    Spaja SHAP važnosti Model A i Model B u jednu tabelu radi direktnog
    poređenja - centralni deo interpretacije (korak 8): kako se važnost
    atributa menja kada se dodaju 360 podaci.
    """
    merged = pd.merge(
        imp_a.rename(columns={"mean_abs_shap": "mean_abs_shap_A"}),
        imp_b.rename(columns={"mean_abs_shap": "mean_abs_shap_B"}),
        on="feature", how="outer",
    ).fillna(0)
    merged["rank_A"] = merged["mean_abs_shap_A"].rank(ascending=False)
    merged["rank_B"] = merged["mean_abs_shap_B"].rank(ascending=False)
    merged["rank_change"] = merged["rank_A"] - merged["rank_B"]
    merged = merged.sort_values("mean_abs_shap_B", ascending=False)

    out_path = TABLES_DIR / "shap_importance_comparison_A_vs_B.csv"
    merged.round(4).to_csv(out_path, index=False)
    print("\n-- Poređenje važnosti atributa: Model A vs Model B --")
    print(merged.head(15).to_string(index=False))
    return merged


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ODDS RATIO (logistička regresija)")
    print("=" * 65)
    odds_ratios_table("model_a", "model_a_lr.csv")
    odds_ratios_table("model_b", "model_b_lr.csv")

    print("\n" + "=" * 65)
    print("SHAP (XGBoost)")
    print("=" * 65)
    imp_a = shap_analysis("model_a_xgboost", "model_a_xgb.csv")
    imp_b = shap_analysis("model_b_xgboost", "model_b_xgb.csv")

    compare_importance_a_vs_b(imp_a, imp_b)
