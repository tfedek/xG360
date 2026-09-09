"""
11_auc_aggregation_reconciliation.py
====================================
Odgovor na recenziju (prof. Pap), tacka 2:

Objasniti razliku AUC = 0,772 (Tabela 3, macro-average LOTO ROC AUC po
turniru) i pooled OOF AUC = 0,709/0,713 (poglavlje o klasterskom bootstrapu),
i pokazati da izotona kalibracija (monotona) NE menja AUC.

Ovo se racuna U ISTOM v2 pipeline-u (LOTO OOF, average_precision tuning) koji
je proizveo broj 0,709 - da bude direktno uporedivo. Racunamo, na ISTIM
LOTO out-of-fold predikcijama:
  - macro AUC  = prosek AUC-ova po held-out turniru
  - pooled AUC = jedan AUC na svim spojenim OOF predikcijama
  - i za NEKALIBRISANE i za KALIBRISANE (izotona, fit na train foldu) predikcije

Oznake nacina agregiranja se eksplicitno cuvaju.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL, RANDOM_STATE,
)
from football_xg.modeling import load_modeling_data, make_models
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "model_training"
ensure_dirs(OUT_DIR)


def prepare_xy(df, numeric_features):
    used = numeric_features + CATEGORICAL + [TARGET, "tournament", "match_id"]
    data = df[used].copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL:
        data[col] = data[col].astype(str)
    data[TARGET] = data[TARGET].astype(int)
    return (data[numeric_features + CATEGORICAL], data[TARGET],
            data["tournament"], data["match_id"])


def collect_oof(df, numeric_features, key):
    """LOTO OOF: nekalibrisane i in-fold izotono kalibrisane predikcije,
    plus po-turniru AUC. Kalibrator fitovan SAMO na trening foldu."""
    X, y, g_tourn, g_match = prepare_xy(df, numeric_features)
    splitter = LeaveOneGroupOut()
    _, (estimator, grid) = list(make_models(numeric_features).items())[
        0 if key == "logistic" else 1]

    oof_u = np.zeros(len(y))
    oof_c = np.zeros(len(y))
    per_u, per_c = {}, {}

    for tr, te in splitter.split(X, y, g_tourn):
        gm_tr = g_match.iloc[tr]
        inner = StratifiedGroupKFold(n_splits=3)
        search = GridSearchCV(estimator, grid, scoring="average_precision",
                               cv=inner, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=gm_tr)
        best = search.best_estimator_

        p_te = best.predict_proba(X.iloc[te])[:, 1]
        p_tr = best.predict_proba(X.iloc[tr])[:, 1]
        oof_u[te] = p_te

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y.iloc[tr].values)
        oof_c[te] = iso.transform(p_te)

        held = g_tourn.iloc[te].iloc[0]
        per_u[held] = roc_auc_score(y.iloc[te], p_te)
        per_c[held] = roc_auc_score(y.iloc[te], iso.transform(p_te))

    return y.values, oof_u, oof_c, per_u, per_c


def main():
    df = load_modeling_data(DATASET_PATH)
    rows = []
    for label, feats in [("Model A", MODEL_A_NUMERIC), ("Model B", MODEL_B_NUMERIC)]:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        y, oof_u, oof_c, per_u, per_c = collect_oof(df, feats, "logistic")

        macro_u = np.mean(list(per_u.values()))
        macro_c = np.mean(list(per_c.values()))
        pooled_u = roc_auc_score(y, oof_u)
        pooled_c = roc_auc_score(y, oof_c)

        print("  AUC po turniru (nekalibrisano): " +
              ", ".join(f"{k}={v:.3f}" for k, v in per_u.items()))
        print(f"  macro  (prosek po turniru):  nekalib={macro_u:.4f}  kalib={macro_c:.4f}")
        print(f"  pooled (svi OOF zajedno):    nekalib={pooled_u:.4f}  kalib={pooled_c:.4f}")
        print(f"  --> kalib - nekalib (pooled): {pooled_c - pooled_u:+.4f}  (izotona monotona -> ~0)")
        print(f"  --> macro - pooled (nekalib): {macro_u - pooled_u:+.4f}  (efekat NACINA agregiranja)")

        rows.append({
            "model": label,
            "macro_nekalibrisano": round(macro_u, 4),
            "macro_kalibrisano": round(macro_c, 4),
            "pooled_nekalibrisano": round(pooled_u, 4),
            "pooled_kalibrisano": round(pooled_c, 4),
            "kalib_minus_nekalib_pooled": round(pooled_c - pooled_u, 4),
            "macro_minus_pooled_nekalib": round(macro_u - pooled_u, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "auc_aggregation_reconciliation.csv", index=False)
    print(f"\nSacuvano: {OUT_DIR / 'auc_aggregation_reconciliation.csv'}")
    print("\nZAKLJUCAK (za rad):")
    print("  1) Kalibrisan vs nekalibrisan (pooled): razlika ~0 -> izotona (monotona)")
    print("     transformacija cuva rang, pa AUC ostaje prakticno isti. Razlika 0,772")
    print("     vs 0,709 NIJE posledica kalibracije.")
    print("  2) macro vs pooled: pooled racuna jedan AUC na spojenim predikcijama iz")
    print("     tri turnira sa razlicitim baznim stopama golova; macro uproseci tri")
    print("     odvojena AUC-a. To su dva razlicita nacina agregiranja i legitimno daju")
    print("     razlicite brojeve. U radu treba EKSPLICITNO oznaciti koji je koji.")


if __name__ == "__main__":
    main()
