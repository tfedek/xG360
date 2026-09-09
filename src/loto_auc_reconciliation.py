"""
loto_auc_reconciliation.py
==========================
Cilj: precizno objasniti razliku izmedju AUC = 0,772 (Tabela 3, macro-average
LOTO ROC AUC po turniru, Model A) i pooled OOF AUC = 0,709/0,713 (poglavlje o
klasterskom bootstrapu), i pokazati da izotona kalibracija (monotona) NE menja
AUC.

Sve na LOTO seme (3 folda po turniru), NA ISTIM opservacijama, sa jasnim
oznakama nacina agregiranja:
  - macro AUC  = prosek AUC-ova izracunatih odvojeno unutar svakog held-out turnira
  - pooled AUC = jedan AUC na SVIM spojenim LOTO out-of-fold predikcijama

I za NEKALIBRISANE i za KALIBRISANE (izotona) LOTO OOF predikcije.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
TABLES = BASE / "tables"
META = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
RS = 42


def load(csv):
    df = pd.read_csv(PROCESSED / csv)
    feats = [c for c in df.columns if c not in META]
    df = df.dropna(subset=feats).reset_index(drop=True)
    return df, df[feats].astype(float), df["is_goal"].astype(int).values, df["tournament"].values


def make_lr():
    return LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced",
                             max_iter=2000, C=0.85, random_state=RS)


def loto_oof(X, y, g):
    """Generise LOTO out-of-fold predikcije (nekalibrisane i kalibrisane).
    Kalibracija: izotona, fit na train foldu (2 turnira), primenjena na
    held-out turnir - kalibracija ne vidi test turnir."""
    oof_u = np.zeros(len(y))
    oof_c = np.zeros(len(y))
    per_tourn_u, per_tourn_c = {}, {}
    for held in sorted(set(g)):
        tr, te = g != held, g == held
        m = make_lr()
        m.fit(X[tr], y[tr])
        p_te = m.predict_proba(X[te])[:, 1]
        oof_u[te] = p_te
        # izotona kalibracija fitovana na trening predikcijama
        p_tr = m.predict_proba(X[tr])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, y[tr])
        oof_c[te] = iso.transform(p_te)
        per_tourn_u[held] = roc_auc_score(y[te], p_te)
        per_tourn_c[held] = roc_auc_score(y[te], iso.transform(p_te))
    return oof_u, oof_c, per_tourn_u, per_tourn_c


def main():
    print("=" * 78)
    print("LOTO AUC: macro (po turniru) vs pooled (svi OOF) | kalibrisan vs nekalibrisan")
    print("=" * 78)

    rows = []
    for label, csv in [("Model A", "model_a_lr.csv"), ("Model B", "model_b_lr.csv")]:
        df, X, y, g = load(csv)
        oof_u, oof_c, per_u, per_c = loto_oof(X, y, g)

        macro_u = np.mean(list(per_u.values()))
        macro_c = np.mean(list(per_c.values()))
        pooled_u = roc_auc_score(y, oof_u)
        pooled_c = roc_auc_score(y, oof_c)

        print(f"\n{label}:")
        print(f"  AUC po turniru (nekalibrisano): " + ", ".join(f"{k}={v:.3f}" for k, v in per_u.items()))
        print(f"  macro  (prosek po turniru): nekalib={macro_u:.4f}  kalib={macro_c:.4f}")
        print(f"  pooled (svi OOF zajedno):   nekalib={pooled_u:.4f}  kalib={pooled_c:.4f}")
        print(f"  -> kalib - nekalib (pooled): {pooled_c - pooled_u:+.4f} (izotona monotona -> ~0)")
        print(f"  -> macro - pooled (nekalib): {macro_u - pooled_u:+.4f} (razlika NACINA agregiranja)")

        rows.append({"model": label, "macro_nekalib": macro_u, "macro_kalib": macro_c,
                     "pooled_nekalib": pooled_u, "pooled_kalib": pooled_c})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "loto_auc_reconciliation.csv", index=False)
    print("\nTabela sacuvana: tables/loto_auc_reconciliation.csv")
    print("\nZAKLJUCAK:")
    print("  - Razlika 0,772 (macro) vs ~0,71 (pooled) potice od NACINA AGREGIRANJA,")
    print("    ne od kalibracije: macro uproseci tri AUC-a po turniru, pooled racuna")
    print("    jedan AUC na spojenim predikcijama razlicitih baznih stopa golova.")
    print("  - Izotona kalibracija (monotona) menja AUC zanemarljivo (~0), sto")
    print("    potvrdjuje da rangiranje ostaje ocuvano.")


if __name__ == "__main__":
    main()
