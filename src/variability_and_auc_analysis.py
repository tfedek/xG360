"""
variability_and_auc_analysis.py
===============================
Dopunske analize varijabilnosti i agregacije AUC vrednosti:

(1) Varijabilnost u tabelama:
    - Stratified (Group) K-Fold: prosek +/- SD preko 5 foldova
    - LOTO: rezultat po SVAKOM turniru pojedinačno + prosek i SD kao
      deskriptivni sažetak (samo 3 turnira)

(2) Objašnjenje razlike AUC (macro-average po turniru) vs pooled OOF AUC:
    - pooled OOF: jedan AUC na SVIM spojenim out-of-fold predikcijama
    - macro: prosek AUC-ova izračunatih odvojeno po turniru
    Računa se pooled OOF AUC i za NEKALIBRISANE i za KALIBRISANE
    (izotona) predikcije na IDENTIČNIM opservacijama, da se pokaže da
    izotona kalibracija (monotona) ne menja rang (AUC ~ isti).

(3) Parni bootstrap interval poverenja razlike Model A vs Model B,
    resamplovanjem PO UTAKMICAMA (cluster bootstrap), koristeći
    out-of-fold predikcije oba modela na istim opservacijama.

Napomena o dizajnu: koristimo StratifiedGroupKFold sa grupama = match_id
za (1) i za generisanje OOF predikcija u (2)/(3), da šutevi iz iste
utakmice ne budu istovremeno u train i test delu (klaster-svesna podela).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression as IsoReg
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
TABLES = BASE / "tables"
TABLES.mkdir(exist_ok=True)

META = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
RS = 42
rng = np.random.default_rng(RS)


def load(csv_name):
    df = pd.read_csv(PROCESSED / csv_name)
    feats = [c for c in df.columns if c not in META]
    df = df.dropna(subset=feats).reset_index(drop=True)
    X = df[feats].astype(float)
    y = df["is_goal"].astype(int).values
    groups = df["tournament"].values
    match_id = df["match_id"].values
    return df, X, y, groups, match_id


def make_lr():
    return LogisticRegression(penalty="l2", solver="lbfgs",
                             class_weight="balanced", max_iter=2000,
                             C=0.85, random_state=RS)


def make_xgb(y):
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                        scale_pos_weight=pw, n_estimators=200, max_depth=3,
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, random_state=RS, n_jobs=-1)


# ==========================================================================
# (1) + (2): OOF predikcije preko StratifiedGroupKFold (grupe = match_id)
#            + macro (po turniru) vs pooled AUC + kalibrisan vs nekalibrisan
# ==========================================================================
def oof_predictions(X, y, match_id, model_factory, n_splits=5):
    """Vraća out-of-fold predikcije (nekalibrisane) za sve opservacije,
    koristeći StratifiedGroupKFold sa grupama = match_id."""
    oof = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    for tr, te in sgkf.split(X, y, groups=match_id):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def isotonic_oof(y, oof_uncal, match_id, n_splits=5):
    """Kalibriše OOF predikcije izotonom regresijom, ali unutar iste
    grupne CV šeme (kalibracija fit-ovana na train foldu, primenjena na
    test fold) - da kalibracija ne 'vidi' test opservacije."""
    cal = np.zeros(len(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    for tr, te in sgkf.split(oof_uncal.reshape(-1, 1), y, groups=match_id):
        iso = IsoReg(out_of_bounds="clip")
        iso.fit(oof_uncal[tr], y[tr])
        cal[te] = iso.transform(oof_uncal[te])
    return cal


def macro_vs_pooled_auc(y, prob, tournaments):
    """Vraća (macro_auc, pooled_auc, per_tournament_dict)."""
    per = {}
    for t in sorted(set(tournaments)):
        mask = tournaments == t
        if len(set(y[mask])) > 1:
            per[t] = roc_auc_score(y[mask], prob[mask])
    macro = np.mean(list(per.values()))
    pooled = roc_auc_score(y, prob)
    return macro, pooled, per


# ==========================================================================
# (1) LOTO po turniru + prosek/SD
# ==========================================================================
def loto_per_tournament(X, y, groups, model_factory):
    rows = []
    for held in sorted(set(groups)):
        tr = groups != held
        te = groups == held
        m = model_factory()
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "held_out": held,
            "roc_auc": roc_auc_score(y[te], prob),
            "pr_auc": average_precision_score(y[te], prob),
            "brier": brier_score_loss(y[te], prob),
            "n_test": int(te.sum()),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# (1) K-Fold prosek +/- SD (grupno, po match_id)
# ==========================================================================
def kfold_mean_sd(X, y, match_id, model_factory, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    rows = []
    for i, (tr, te) in enumerate(sgkf.split(X, y, groups=match_id), 1):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        rows.append({
            "fold": i,
            "roc_auc": roc_auc_score(y[te], prob),
            "pr_auc": average_precision_score(y[te], prob),
            "brier": brier_score_loss(y[te], prob),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# (3) Parni cluster bootstrap CI razlike AUC (Model B - Model A),
#     resampling PO UTAKMICAMA na zajedničkim OOF predikcijama
# ==========================================================================
def paired_cluster_bootstrap(y, prob_a, prob_b, match_id, n_boot=2000):
    """Resamplira match_id sa zamenom; za svaki bootstrap uzorak računa
    AUC_B - AUC_A na istim (uparenim) opservacijama. Vraća (obs_diff, lo, hi,
    p_approx)."""
    obs = roc_auc_score(y, prob_b) - roc_auc_score(y, prob_a)
    uniq = np.unique(match_id)
    # mapiranje match_id -> indeksi
    idx_by_match = {m: np.where(match_id == m)[0] for m in uniq}
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in sampled])
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        d = roc_auc_score(yb, prob_b[idx]) - roc_auc_score(yb, prob_a[idx])
        diffs.append(d)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # dvosmerni aproksimativni p: udeo bootstrap razlika koje prelaze 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, p, len(diffs)


def main():
    print("=" * 78)
    print("DOPUNSKE ANALIZE - varijabilnost, pooled OOF AUC, parni bootstrap")
    print("=" * 78)

    # --- Učitavanje (LR varijante za glavnu analizu, kako u radu) ---
    df_a, Xa, ya, ga, ma = load("model_a_lr.csv")
    df_b, Xb, yb, gb, mb = load("model_b_lr.csv")

    # Model A i B mogu imati različit broj redova (B izbacuje 2 NaN reda).
    # Za PARNO poređenje radimo na PRESEKU (isti match_id + isti indeks reda).
    # Poravnanje preko (match_id, player, minute-ekvivalent) je složeno; umesto
    # toga radimo presek po indeksu iz shots_raw redosleda: jednostavnije je
    # ponovo učitati zajednički skup. Ovde koristimo činjenicu da su oba
    # izvedena iz istog shots_raw - poravnavamo po (match_id, statsbomb_xg,
    # is_goal) tuple kao ključu.
    key_a = df_a[["match_id", "statsbomb_xg", "is_goal"]].round({"statsbomb_xg": 6})
    key_b = df_b[["match_id", "statsbomb_xg", "is_goal"]].round({"statsbomb_xg": 6})
    key_a["_ka"] = key_a.astype(str).agg("|".join, axis=1)
    key_b["_kb"] = key_b.astype(str).agg("|".join, axis=1)
    common = set(key_a["_ka"]) & set(key_b["_kb"])
    print(f"\nModel A n={len(df_a)}, Model B n={len(df_b)}, presek={len(common)}")

    mask_a = key_a["_ka"].isin(common).values
    mask_b = key_b["_kb"].isin(common).values
    # sortiramo oba na isti redosled ključa da budu uparene opservacije
    order_a = np.argsort(key_a.loc[mask_a, "_ka"].values)
    order_b = np.argsort(key_b.loc[mask_b, "_kb"].values)

    Xa2 = Xa[mask_a].iloc[order_a].reset_index(drop=True)
    ya2 = ya[mask_a][order_a]
    ga2 = ga[mask_a][order_a]
    ma2 = ma[mask_a][order_a]
    Xb2 = Xb[mask_b].iloc[order_b].reset_index(drop=True)
    yb2 = yb[mask_b][order_b]
    mb2 = mb[mask_b][order_b]
    assert (ya2 == yb2).all(), "poravnanje ishoda nije uspelo"
    assert (ma2 == mb2).all(), "poravnanje match_id nije uspelo"

    # =====================================================================
    # (2) OOF predikcije + macro vs pooled + kalibrisan vs nekalibrisan
    # =====================================================================
    print("\n" + "-" * 78)
    print("(2) POOLED OOF AUC vs MACRO (po turniru) - Model B (logreg), StratGroupKFold")
    print("-" * 78)
    oof_b = oof_predictions(Xb2, yb2, mb2, make_lr)
    oof_b_cal = isotonic_oof(yb2, oof_b, mb2)

    macro_u, pooled_u, per_u = macro_vs_pooled_auc(yb2, oof_b, ga2)
    macro_c, pooled_c, per_c = macro_vs_pooled_auc(yb2, oof_b_cal, ga2)

    print(f"  NEKALIBRISANO:  macro(po turniru) AUC = {macro_u:.4f} | pooled OOF AUC = {pooled_u:.4f}")
    print(f"  KALIBRISANO:    macro(po turniru) AUC = {macro_c:.4f} | pooled OOF AUC = {pooled_c:.4f}")
    print(f"  Razlika kalib-nekalib (pooled): {pooled_c - pooled_u:+.4f}  (izotona je monotona -> ocekivano ~0)")
    print(f"  AUC po turniru (nekalibrisano): " + ", ".join(f"{k}={v:.3f}" for k,v in per_u.items()))

    pd.DataFrame([
        {"agregacija": "macro (prosek po turniru)", "nekalibrisano": macro_u, "kalibrisano": macro_c},
        {"agregacija": "pooled (svi OOF zajedno)", "nekalibrisano": pooled_u, "kalibrisano": pooled_c},
    ]).to_csv(TABLES / "auc_macro_vs_pooled.csv", index=False)

    # =====================================================================
    # (1) K-Fold prosek +/- SD i LOTO po turniru - za oba modela
    # =====================================================================
    print("\n" + "-" * 78)
    print("(1) K-FOLD prosek +/- SD (StratGroupKFold, grupe=match_id)")
    print("-" * 78)
    for name, X, y, mid in [("Model A", Xa2, ya2, ma2), ("Model B", Xb2, yb2, mb2)]:
        kf = kfold_mean_sd(X, y, mid, make_lr)
        m, s = kf.mean(numeric_only=True), kf.std(numeric_only=True)
        print(f"  {name}: ROC AUC={m['roc_auc']:.3f}+/-{s['roc_auc']:.3f} | "
              f"PR AUC={m['pr_auc']:.3f}+/-{s['pr_auc']:.3f} | "
              f"Brier={m['brier']:.4f}+/-{s['brier']:.4f}")
        kf.to_csv(TABLES / f"kfold_variability_{name.replace(' ','_').lower()}.csv", index=False)

    print("\n" + "-" * 78)
    print("(1) LOTO po turniru + prosek/SD")
    print("-" * 78)
    for name, X, y, g in [("Model A", Xa2, ya2, ga2), ("Model B", Xb2, yb2, gb if False else ga2)]:
        lt = loto_per_tournament(X, y, g, make_lr)
        print(f"  {name}:")
        for _, r in lt.iterrows():
            print(f"    {r['held_out']:>9}: ROC AUC={r['roc_auc']:.3f}, PR AUC={r['pr_auc']:.3f}, "
                  f"Brier={r['brier']:.4f} (n={int(r['n_test'])})")
        m, s = lt.mean(numeric_only=True), lt.std(numeric_only=True)
        print(f"    {'prosek':>9}: ROC AUC={m['roc_auc']:.3f}+/-{s['roc_auc']:.3f}, "
              f"PR AUC={m['pr_auc']:.3f}+/-{s['pr_auc']:.3f}, Brier={m['brier']:.4f}+/-{s['brier']:.4f}")
        lt.to_csv(TABLES / f"loto_pertournament_{name.replace(' ','_').lower()}.csv", index=False)

    # =====================================================================
    # (3) Parni cluster bootstrap CI razlike AUC (B - A), po utakmicama
    # =====================================================================
    print("\n" + "-" * 78)
    print("(3) PARNI CLUSTER BOOTSTRAP CI razlike ROC AUC (Model B - Model A)")
    print("    resampling po UTAKMICAMA (match_id), OOF predikcije, iste opservacije")
    print("-" * 78)
    oof_a = oof_predictions(Xa2, ya2, ma2, make_lr)
    obs, lo, hi, p, n_ok = paired_cluster_bootstrap(ya2, oof_a, oof_b, ma2, n_boot=2000)
    print(f"  Opazena razlika AUC (B - A): {obs:+.4f}")
    print(f"  95% CI (cluster bootstrap, {n_ok} validnih iteracija): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Aproksimativni dvosmerni p: {p:.4f}")
    print(f"  CI {' NE sadrzi 0 -> znacajno' if lo > 0 or hi < 0 else 'sadrzi 0 -> nije znacajno'}")

    pd.DataFrame([{
        "opazena_razlika_AUC_B_minus_A": obs,
        "ci_lo_95": lo, "ci_hi_95": hi, "p_approx": p,
        "bootstrap_iteracija": n_ok, "resampling_jedinica": "match_id",
        "predikcije": "out-of-fold (StratifiedGroupKFold, grupe=match_id)",
    }]).to_csv(TABLES / "paired_cluster_bootstrap_auc.csv", index=False)

    print("\nGotovo. Tabele sacuvane u tables/.")


if __name__ == "__main__":
    main()
