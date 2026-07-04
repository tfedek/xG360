"""
assumption_checks.py
======================
Korak 3 (deo: provera pretpostavki) metodologije.

Provera pretpostavki logističke regresije PRE treniranja finalnih modela:
  1. Multikolinearnost (VIF)
  2. Linearnost logita (Box-Tidwell test za numeričke prediktore)
  3. Class imbalance (već poznato iz EDA: ~9.4% golova -> odnos ~9.7:1)
  4. Broj događaja (golova) po prediktoru (rule of thumb: 10-20 per predictor)

Pokretanje: python3 assumption_checks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"

NON_FEATURE_COLS = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Računa Variance Inflation Factor za dati skup numeričkih kolona.
    Pravilo: VIF > 5 (neki autori koriste 10) signalizira problematičnu
    multikolinearnost koju treba rešiti (izbacivanjem ili transformacijom
    jedne od korelisanih promenljivih).
    """
    X = df[numeric_cols].dropna().astype(float)
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data[vif_data["feature"] != "const"].sort_values("VIF", ascending=False)


def box_tidwell_check(df: pd.DataFrame, numeric_col: str, target: str = "is_goal") -> dict:
    """
    Pojednostavljena verzija Box-Tidwell testa: dodaje interakcioni član
    x * ln(x) u logističku regresiju i testira njegovu p-vrednost.
    Značajan koeficijent (p < 0.05) ukazuje na narušenu linearnost logita
    za tu promenljivu - signal da treba transformacija (npr. log, polinom)
    pre uključivanja u finalni model.
    """
    sub = df[[numeric_col, target]].dropna().copy()
    sub = sub[sub[numeric_col] > 0]  # ln zahteva pozitivne vrednosti
    sub["x_lnx"] = sub[numeric_col] * np.log(sub[numeric_col])

    X = sm.add_constant(sub[[numeric_col, "x_lnx"]])
    y = sub[target]
    model = sm.Logit(y, X).fit(disp=0)

    pval = model.pvalues["x_lnx"]
    return {
        "variable": numeric_col,
        "interaction_pvalue": round(pval, 4),
        "linearity_violated": bool(pval < 0.05),
        "n_obs": len(sub),
    }


def events_per_predictor(df: pd.DataFrame, target: str = "is_goal") -> dict:
    """Rule of thumb provera: 10-20 događaja (manjinske klase) po prediktoru."""
    n_predictors = len(_feature_cols(df))
    n_events = int(df[target].sum())
    ratio = round(n_events / n_predictors, 1) if n_predictors else np.nan
    return {
        "n_predictors": n_predictors,
        "n_events_minority_class": n_events,
        "events_per_predictor": ratio,
        "adequate": bool(ratio >= 10),
    }


def class_imbalance_summary(df: pd.DataFrame, target: str = "is_goal") -> dict:
    rate = df[target].mean()
    return {
        "n_total": len(df),
        "n_positive": int(df[target].sum()),
        "positive_rate": round(rate, 4),
        "imbalance_ratio": round((1 - rate) / rate, 2),
        "severity": "umerena (ne zahteva resampling po default-u; "
                    "preporučeno: class_weight='balanced' + fokus na PR-AUC)"
                    if 0.05 < rate < 0.35 else
                    "ekstremna (razmotriti resampling unutar CV foldova)",
    }


def run_all_checks(model_label: str, csv_name: str) -> dict:
    df = pd.read_csv(PROCESSED_DIR / csv_name)
    print(f"\n{'='*60}\nProvera pretpostavki: {model_label}\n{'='*60}")

    # --- 1. VIF ---
    # Numeričke kolone identifikujemo dinamički (sve float/int osim is_goal/statsbomb_xg
    # i osim binarnih dummy/0-1 kolona koje VIF tretira drugačije, ali ih ovde
    # ostavljamo unutra jer su deo istog modela - standardna praksa).
    numeric_like = [c for c in _feature_cols(df)
                     if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().all()]
    vif_df = compute_vif(df, numeric_like)
    print("\n-- VIF (Variance Inflation Factor) --")
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df["VIF"] > 5]
    if len(high_vif):
        print(f"\n  UPOZORENJE: {len(high_vif)} promenljivih sa VIF > 5: "
              f"{', '.join(high_vif['feature'].tolist())}")

    # --- 2. Linearnost logita (Box-Tidwell) na ključnim kontinuiranim promenljivama ---
    print("\n-- Linearnost logita (Box-Tidwell aproksimacija) --")
    bt_results = []
    for col in ["distance_to_goal", "shot_angle_deg"]:
        if col in df.columns:
            res = box_tidwell_check(df, col)
            bt_results.append(res)
            flag = "NARUŠENA" if res["linearity_violated"] else "OK"
            print(f"  {col}: p={res['interaction_pvalue']} -> linearnost {flag}")

    # --- 3. Class imbalance ---
    imb = class_imbalance_summary(df)
    print(f"\n-- Class imbalance -- {imb['n_positive']}/{imb['n_total']} "
          f"({100*imb['positive_rate']:.1f}%), odnos {imb['imbalance_ratio']}:1 -> {imb['severity']}")

    # --- 4. Broj događaja po prediktoru ---
    epp = events_per_predictor(df)
    status = "DOVOLJNO" if epp["adequate"] else "GRANIČNO/NEDOVOLJNO"
    print(f"\n-- Broj događaja po prediktoru -- {epp['n_events_minority_class']} golova / "
          f"{epp['n_predictors']} prediktora = {epp['events_per_predictor']} ({status})")

    return {
        "vif": vif_df, "box_tidwell": bt_results,
        "class_imbalance": imb, "events_per_predictor": epp,
    }


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results_a = run_all_checks("Model A (klasični atributi)", "model_a.csv")
    results_a["vif"].to_csv(TABLES_DIR / "vif_model_a.csv", index=False)

    results_b = run_all_checks("Model B (klasični + 360 atributi)", "model_b.csv")
    results_b["vif"].to_csv(TABLES_DIR / "vif_model_b.csv", index=False)
