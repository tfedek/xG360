"""
train_models.py
==================
Korak 5 (razvoj prediktivnih modela) i korak 6 (validacija) metodologije.

Za Model A i Model B treniramo:
  - Logistička regresija (na LR varijanti feature seta - log-transformisane
    distance/angle, rešena multikolinearnost)
  - XGBoost (na XGB varijanti - sirove vrednosti, tuning hiperparametara
    putem RandomizedSearchCV sa unutrašnjom (inner) Stratified K-Fold CV)

Validacija u dva nivoa, namerno odvojena:
  (a) Stratified K-Fold (5 foldova) - "koliko model radi na podacima iz
      iste distribucije" - koristi se i za tuning hiperparametara.
  (b) Leave-One-Tournament-Out (LOTO, 3 folda: WC2022/EURO2020/EURO2024)
      - "koliko model generalizuje na potpuno nov turnir" - test skup u
      svakom foldu NIKADA ne učestvuje u tuning-u (sprečava data leakage).

Pokretanje: python3 train_models.py
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "fitted_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
RANDOM_STATE = 42


def _xy_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Vraća (X, y, tournament_labels) - poslednje za LOTO grupisanje.

    Napomena: nekoliko (vrlo malo, <0.1% uzorka) redova ima rezidualni NaN
    u open_goal_angle_ratio - ekstremni geometrijski edge-case (shot_angle_deg
    == 0, deljenje nulom u formuli otvorenosti ugla). Ti redovi se izbacuju
    listwise (potpuni case-by-case izostavljanje), što je opravdano s
    obzirom na zanemarljiv udeo i odsustvo sistematskog obrasca u tome koji
    redovi su pogođeni (nasumičan geometrijski ekstrem, ne MNAR uzorak)."""
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df_clean = df.dropna(subset=feature_cols)
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        print(f"  [napomena] izbačeno {n_dropped} redova sa rezidualnim NaN "
              f"u feature kolonama ({100*n_dropped/len(df):.2f}% uzorka)")
    X = df_clean[feature_cols].astype(float)
    y = df_clean["is_goal"].astype(int)
    groups = df_clean["tournament"]
    return X, y, groups


def _evaluate(y_true, y_prob, y_pred) -> dict:
    """Standardni set metrika diskriminacije i kalibracije (korak 7)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": len(y_true),
    }


def tune_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    RandomizedSearchCV za regularizacionu jačinu (C) logističke regresije,
    sa unutrašnjom Stratified 5-Fold CV. class_weight='balanced' korišćen
    po odluci iz koraka provere pretpostavki (umerena neuravnoteženost,
    ponderisanje umesto resampling-a).
    """
    param_dist = {"C": np.logspace(-3, 2, 30)}
    base = LogisticRegression(
        penalty="l2", solver="lbfgs", class_weight="balanced",
        max_iter=2000, random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=15, scoring="average_precision",
        cv=inner_cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def tune_xgboost(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """
    RandomizedSearchCV za XGBoost hiperparametre, unutrašnja Stratified
    5-Fold CV. scale_pos_weight kompenzuje class imbalance (analogno
    class_weight='balanced' kod logističke regresije).
    """
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
    }
    base = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
        use_label_encoder=False, n_jobs=-1,
    )
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=25, scoring="average_precision",
        cv=inner_cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def run_stratified_kfold(model_name: str, X: pd.DataFrame, y: pd.Series,
                          tuner_fn, n_splits: int = 5) -> dict:
    """
    Stratified K-Fold validacija: u SVAKOM foldu se hiperparametri ponovo
    podešavaju (tuner_fn) SAMO na trening delu tog folda, zatim se
    procenjuju metrike na test delu - sprečava curenje informacija iz
    test dela u tuning unutar ovog koraka.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model, _ = tuner_fn(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = _evaluate(y_test, y_prob, y_pred)
        metrics["fold"] = fold_i
        fold_results.append(metrics)
        print(f"    [{model_name}] K-Fold {fold_i}/{n_splits}: "
              f"ROC AUC={metrics['roc_auc']:.3f}, PR AUC={metrics['pr_auc']:.3f}, "
              f"Brier={metrics['brier_score']:.4f}")

    df_results = pd.DataFrame(fold_results)
    return {"per_fold": df_results, "mean": df_results.mean(numeric_only=True),
            "std": df_results.std(numeric_only=True)}


def run_leave_one_tournament_out(model_name: str, X: pd.DataFrame, y: pd.Series,
                                  groups: pd.Series, tuner_fn) -> dict:
    """
    LOTO: za svaki turnir T, trening se radi na PREOSTALA DVA turnira
    (uklj. unutrašnji tuning SAMO na tom trening skupu), test se radi na
    T. Test turnir nikada nije "viđen" tokom tuning-a - prava procena
    generalizacije na potpuno nov kontekst.
    """
    tournaments = sorted(groups.unique())
    fold_results = []
    for held_out in tournaments:
        train_mask = groups != held_out
        test_mask = groups == held_out

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model, best_params = tuner_fn(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = _evaluate(y_test, y_prob, y_pred)
        metrics["held_out_tournament"] = held_out
        fold_results.append(metrics)
        print(f"    [{model_name}] LOTO held-out={held_out}: "
              f"ROC AUC={metrics['roc_auc']:.3f}, PR AUC={metrics['pr_auc']:.3f}, "
              f"Brier={metrics['brier_score']:.4f}  (n_test={metrics['n']})")

    df_results = pd.DataFrame(fold_results)
    return {"per_fold": df_results, "mean": df_results.mean(numeric_only=True),
            "std": df_results.std(numeric_only=True)}


def fit_final_model(model_name: str, X: pd.DataFrame, y: pd.Series, tuner_fn):
    """Trenira finalni model na CELOM dostupnom skupu (za SHAP/interpretaciju
    i za buduću upotrebu u demonstracionoj aplikaciji), čuva na disk."""
    model, best_params = tuner_fn(X, y)
    joblib.dump({"model": model, "feature_names": list(X.columns), "params": best_params},
                MODELS_DIR / f"{model_name}.joblib")
    print(f"  Finalni model '{model_name}' sačuvan. Najbolji parametri: {best_params}")
    return model, best_params


def run_full_pipeline():
    specs = {
        "model_a_logreg": ("model_a_lr.csv", tune_logistic_regression),
        "model_a_xgboost": ("model_a_xgb.csv", tune_xgboost),
        "model_b_logreg": ("model_b_lr.csv", tune_logistic_regression),
        "model_b_xgboost": ("model_b_xgb.csv", tune_xgboost),
    }

    all_kfold = {}
    all_loto = {}

    for model_name, (csv_name, tuner_fn) in specs.items():
        print(f"\n{'='*65}\n{model_name}\n{'='*65}")
        df = pd.read_csv(PROCESSED_DIR / csv_name)
        X, y, groups = _xy_split(df)

        print(f"  n={len(df)}, n_golova={y.sum()}, n_feature={X.shape[1]}")

        print("  -- Stratified K-Fold (5 foldova) --")
        kfold_res = run_stratified_kfold(model_name, X, y, tuner_fn)
        all_kfold[model_name] = kfold_res

        print("  -- Leave-One-Tournament-Out --")
        loto_res = run_leave_one_tournament_out(model_name, X, y, groups, tuner_fn)
        all_loto[model_name] = loto_res

        print("  -- Treniranje finalnog modela (ceo skup, za SHAP/app) --")
        fit_final_model(model_name, X, y, tuner_fn)

    # Eksport sažetih tabela za rad
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    kfold_summary = pd.DataFrame({name: res["mean"] for name, res in all_kfold.items()}).T
    loto_summary = pd.DataFrame({name: res["mean"] for name, res in all_loto.items()}).T
    kfold_summary.to_csv(TABLES_DIR / "kfold_summary.csv")
    loto_summary.to_csv(TABLES_DIR / "loto_summary.csv")

    for name, res in all_kfold.items():
        res["per_fold"].to_csv(TABLES_DIR / f"kfold_perfold_{name}.csv", index=False)
    for name, res in all_loto.items():
        res["per_fold"].to_csv(TABLES_DIR / f"loto_perfold_{name}.csv", index=False)

    print("\n\n========== SAŽETAK: Stratified K-Fold (mean preko 5 foldova) ==========")
    print(kfold_summary[["roc_auc", "pr_auc", "f1", "brier_score"]].round(3))
    print("\n========== SAŽETAK: Leave-One-Tournament-Out (mean preko 3 turnira) ==========")
    print(loto_summary[["roc_auc", "pr_auc", "f1", "brier_score"]].round(3))

    return all_kfold, all_loto


if __name__ == "__main__":
    run_full_pipeline()
