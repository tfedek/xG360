"""
significance_tests.py
========================
Dopuna metodologije (koraci 6-8): formalni statistički testovi da se
potvrdi da je razlika Model A vs Model B statistički značajna, ne samo
narativno dosledna kroz metrike.

Sadrži tri analize, urađene na stvarnim podacima iz pipeline-a:

  (a) Likelihood Ratio (LR) test - logistička regresija, Model A vs B.
      Model A je formalno ugnježden u Model B (24 feature-a Model A su
      tačan podskup 31 feature-a Model B; razlika su 7 StatsBomb 360
      atributa: n_defenders_in_cone, n_teammates_in_cone, n_opponents_close,
      open_goal_angle_ratio, nearest_defender_to_shot_line, pressure_score,
      goalkeeper_anomaly). LR test je formalno ispravan test ugnježdenih
      modela: LR = 2*(llf_B - llf_A) ~ chi2(df), df = broj dodatnih
      parametara.

  (b) Bootstrap 95% CI za razliku ROC AUC (Model B - Model A), XGBoost
      varijanta, na test skupu jednog train/test splita (80/20, isti
      protokol kao evaluation_plots.py). XGBoost nije ugnježden
      parametarski model na isti način kao logistička regresija, pa LR
      test ne važi; bootstrap (resampling test skupa sa replacement,
      2000 iteracija) je standardna alternativa.

  (c) Tačan broj redova izbačenih u _xy_split (train_models.py) zbog
      rezidualnog NaN u open_goal_angle_ratio (edge-case shot_angle_deg==0).

NAPOMENA O METODOLOŠKOJ DOSLEDNOSTI: i (a) i (b) su urađeni na
ZAJEDNIČKOM uzorku šuteva (3.966, ne 3.968) - oni isti redovi koje
Model B mora da izbaci zbog NaN u 360 atributima, izbačeni su i iz
Model A skupa za ovo poređenje, da bi oba modela bila fitovana/testirana
na identičnom skupu opservacija (preduslov za validan LR test, i
preduslov za paired bootstrap na XGBoost AUC razlici).

Brojke u (b) su iz JEDNOG 80/20 train/test splita (isti protokol kao
evaluation_plots.py), NE iz Leave-One-Tournament-Out validacije. To su
različiti konteksti i ne treba ih mešati: LOTO brojke (Model A 0,772 /
Model B 0,792 ROC AUC, prosek kroz 3 turnira) ostaju glavni rezultat
validacije generalizacije u poglavlju 5.3 rada; bootstrap CI ovde je
DODATNA, formalna potvrda značajnosti na nivou jednog reprezentativnog
test skupa, što je metodološki ispravniji opis nego mešanje ta dva broja.

Pokretanje: python3 significance_tests.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
META_COLS = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
RANDOM_STATE = 42

BEST_PARAMS_XGB = {
    "n_estimators": 200, "max_depth": 2, "learning_rate": 0.05,
    "subsample": 1.0, "colsample_bytree": 0.8, "min_child_weight": 1,
}


def _load_raw(csv_name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / csv_name)


def count_dropped_rows() -> dict:
    print("=" * 65)
    print("(c) Broj redova izbačenih zbog rezidualnog NaN (_xy_split)")
    print("=" * 65)
    results = {}
    for csv_name in ["model_a_lr.csv", "model_b_lr.csv", "model_a_xgb.csv", "model_b_xgb.csv"]:
        df = _load_raw(csv_name)
        feature_cols = [c for c in df.columns if c not in META_COLS]
        n_before = len(df)
        n_after = len(df.dropna(subset=feature_cols))
        n_dropped = n_before - n_after
        results[csv_name] = {"n_before": n_before, "n_after": n_after, "n_dropped": n_dropped}
        print(f"  {csv_name}: {n_before} -> {n_after} ({n_dropped} izbačeno)")
    print()
    return results


def likelihood_ratio_test() -> dict:
    print("=" * 65)
    print("(a) Likelihood Ratio test: Model A (ugnježden) vs Model B")
    print("=" * 65)

    df_a = _load_raw("model_a_lr.csv")
    df_b = _load_raw("model_b_lr.csv")

    feat_a = [c for c in df_a.columns if c not in META_COLS]
    feat_b = [c for c in df_b.columns if c not in META_COLS]

    assert set(feat_a).issubset(set(feat_b)), "Model A mora biti formalno ugnježden u Model B!"
    extra_features = sorted(set(feat_b) - set(feat_a))
    print(f"  Model A: {len(feat_a)} feature-a")
    print(f"  Model B: {len(feat_b)} feature-a")
    print(f"  Dodatni (360) feature-i u B ({len(extra_features)}): {extra_features}")

    df_b_clean = df_b.dropna(subset=feat_b)
    common_idx = df_b_clean.index
    df_a_clean = df_a.loc[common_idx]

    y = df_a_clean["is_goal"].astype(int)
    assert (df_a_clean["is_goal"].values == df_b_clean["is_goal"].values).all()

    X_a = sm.add_constant(df_a_clean[feat_a].astype(float))
    X_b = sm.add_constant(df_b_clean[feat_b].astype(float))

    print(f"\n  N (zajednički uzorak, nakon uklanjanja NaN): {len(y)}")
    print(f"  Golova: {y.sum()} ({100*y.mean():.2f}%)")

    model_a = sm.Logit(y, X_a).fit(disp=0, maxiter=200)
    model_b = sm.Logit(y, X_b).fit(disp=0, maxiter=200)

    llf_a, llf_b = model_a.llf, model_b.llf
    df_diff = X_b.shape[1] - X_a.shape[1]
    lr_stat = 2 * (llf_b - llf_a)
    p_value = stats.chi2.sf(lr_stat, df_diff)

    print(f"\n  log-likelihood Model A: {llf_a:.4f}")
    print(f"  log-likelihood Model B: {llf_b:.4f}")
    print(f"  df (broj dodatnih parametara): {df_diff}")
    print(f"  LR statistika: {lr_stat:.4f}")
    print(f"  p-vrednost (chi2, df={df_diff}): {p_value:.4e}")
    print(f"\n  AIC: Model A={model_a.aic:.2f}, Model B={model_b.aic:.2f}")
    print(f"  BIC: Model A={model_a.bic:.2f}, Model B={model_b.bic:.2f}")
    print(f"  Zakljucak: {'Model B znacajno bolji fit (p<0.05)' if p_value < 0.05 else 'Nema znacajne razlike'}")
    print()

    return {
        "n_obs": int(len(y)), "n_goals": int(y.sum()),
        "extra_features": extra_features,
        "llf_a": float(llf_a), "llf_b": float(llf_b),
        "df_diff": int(df_diff), "lr_statistic": float(lr_stat), "p_value": float(p_value),
        "aic_a": float(model_a.aic), "aic_b": float(model_b.aic),
        "bic_a": float(model_a.bic), "bic_b": float(model_b.bic),
    }


def bootstrap_auc_difference(n_boot: int = 2000) -> dict:
    print("=" * 65)
    print("(b) Bootstrap 95% CI za razliku ROC AUC (XGBoost, B - A)")
    print("=" * 65)

    df_a = _load_raw("model_a_xgb.csv")
    df_b = _load_raw("model_b_xgb.csv")
    feat_a = [c for c in df_a.columns if c not in META_COLS]
    feat_b = [c for c in df_b.columns if c not in META_COLS]

    df_b_clean = df_b.dropna(subset=feat_b)
    common_idx = df_b_clean.index
    df_a_clean = df_a.loc[common_idx]

    y_common = df_a_clean["is_goal"].astype(int)
    assert (df_a_clean["is_goal"].values == df_b_clean["is_goal"].values).all()

    X_a = df_a_clean[feat_a].astype(float).reset_index(drop=True)
    X_b = df_b_clean[feat_b].astype(float).reset_index(drop=True)
    y = y_common.reset_index(drop=True)

    print(f"  Zajednicki uzorak: {len(y)} suteva")

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    Xa_tr, Xa_te = X_a.iloc[idx_train], X_a.iloc[idx_test]
    Xb_tr, Xb_te = X_b.iloc[idx_train], X_b.iloc[idx_test]
    y_tr, y_te = y.iloc[idx_train], y.iloc[idx_test]

    print(f"  Test skup: {len(y_te)} suteva, {y_te.sum()} golova")

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model_a = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             scale_pos_weight=pos_weight, random_state=RANDOM_STATE, **BEST_PARAMS_XGB)
    model_a.fit(Xa_tr, y_tr)
    prob_a_test = model_a.predict_proba(Xa_te)[:, 1]

    model_b = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             scale_pos_weight=pos_weight, random_state=RANDOM_STATE, **BEST_PARAMS_XGB)
    model_b.fit(Xb_tr, y_tr)
    prob_b_test = model_b.predict_proba(Xb_te)[:, 1]

    auc_a = roc_auc_score(y_te, prob_a_test)
    auc_b = roc_auc_score(y_te, prob_b_test)
    observed_diff = auc_b - auc_a

    print(f"\n  ROC AUC Model A: {auc_a:.4f}")
    print(f"  ROC AUC Model B: {auc_b:.4f}")
    print(f"  Opazena razlika (B - A): {observed_diff:.4f}")

    rng = np.random.default_rng(RANDOM_STATE)
    y_te_arr = y_te.values
    n_test = len(y_te_arr)
    boot_diffs = []

    for _ in range(n_boot):
        boot_idx = rng.choice(n_test, size=n_test, replace=True)
        y_boot = y_te_arr[boot_idx]
        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue
        auc_a_boot = roc_auc_score(y_boot, prob_a_test[boot_idx])
        auc_b_boot = roc_auc_score(y_boot, prob_b_test[boot_idx])
        boot_diffs.append(auc_b_boot - auc_a_boot)

    boot_diffs = np.array(boot_diffs)
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    p_approx = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

    print(f"\n  Uspesnih bootstrap iteracija: {len(boot_diffs)}/{n_boot}")
    print(f"  Prosecna bootstrap razlika: {boot_diffs.mean():.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI sadrzi 0: {'DA (nije znacajno)' if ci_lower <= 0 <= ci_upper else 'NE (znacajno)'}")
    print(f"  Priblizna p-vrednost (dvostrana, iz bootstrap distribucije): {p_approx:.4f}")
    print()

    return {
        "n_test": int(n_test), "n_goals_test": int(y_te.sum()),
        "auc_a": float(auc_a), "auc_b": float(auc_b), "observed_diff": float(observed_diff),
        "n_boot_successful": int(len(boot_diffs)),
        "boot_mean_diff": float(boot_diffs.mean()),
        "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
        "p_value_approx": float(p_approx),
    }


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dropped = count_dropped_rows()
    lr_results = likelihood_ratio_test()
    boot_results = bootstrap_auc_difference()

    summary = pd.DataFrame([{
        "test": "Likelihood Ratio (LogReg, A vs B)",
        "statistic": round(lr_results["lr_statistic"], 4),
        "df_or_n": lr_results["df_diff"],
        "p_value": lr_results["p_value"],
        "significant_at_0.05": lr_results["p_value"] < 0.05,
    }, {
        "test": "Bootstrap CI razlike AUC (XGBoost, B-A)",
        "statistic": round(boot_results["observed_diff"], 4),
        "df_or_n": boot_results["n_boot_successful"],
        "p_value": boot_results["p_value_approx"],
        "significant_at_0.05": not (boot_results["ci_lower"] <= 0 <= boot_results["ci_upper"]),
    }])
    summary.to_csv(TABLES_DIR / "significance_tests_summary.csv", index=False)

    print("=" * 65)
    print("SAZETAK (sacuvano u tables/significance_tests_summary.csv)")
    print("=" * 65)
    print(summary.to_string(index=False))
