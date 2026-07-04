"""
evaluation_plots.py
======================
Korak 7 (evaluacija diskriminacije i kalibracije) - grafički deo.

Koristi held-out predikcije iz JEDNOG representativnog train/test splita
(80/20, stratifikovano) - ne iz K-Fold/LOTO petlje (one već imaju svoje
sažete metrike u tables/kfold_summary.csv i loto_summary.csv) - ovde je
cilj VIZUELNI prikaz oblika kalibracije i ROC/PR krivih, što zahteva
jedan konkretan skup predikcija, ne prosek kroz foldove.

Generiše:
  - figures/calibration_curves.png (Model A vs B, LogReg i XGBoost)
  - figures/roc_curves.png
  - figures/pr_curves.png
  - figures/confusion_matrices.png (2x2 grid: A/B x LogReg/XGBoost)

Pokretanje: python3 evaluation_plots.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
META_COLS = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
RANDOM_STATE = 42

# Najbolji hiperparametri preuzeti iz train_models.py tuning rezultata
# (ponovni fit na jednom train/test splitu radi grafičkog prikaza;
# ne radimo ponovni RandomizedSearchCV ovde - to je već urađeno i
# rezultati su zabeleženi u tables/, fokus ovog modula je VIZUELIZACIJA)
BEST_PARAMS_LR = {"C": 0.85, "class_weight": "balanced", "max_iter": 2000}
BEST_PARAMS_XGB = {
    "n_estimators": 200, "max_depth": 2, "learning_rate": 0.05,
    "subsample": 1.0, "colsample_bytree": 0.8, "min_child_weight": 1,
}


def _load_split(csv_name: str):
    df = pd.read_csv(PROCESSED_DIR / csv_name)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].astype(float)
    y = df["is_goal"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def get_test_predictions() -> dict:
    """Trenira po jedan LogReg i XGBoost model za Model A i Model B na
    jednom 80/20 splitu, vraća (y_test, y_prob) za svaku od 4 kombinacije.

    Dodatno: za SVAKI model trenira se i kalibrisana verzija (izotona
    regresija, fit na DODATNOJ unutrašnjoj podeli trening skupa - 'cv=5'
    u CalibratedClassifierCV - tako da kalibracija ne 'vidi' isti trening
    deo na kome je baferni model fitovan, što bi bilo cirkularno).
    Razlog za post-hoc kalibraciju: vizuelna provera (calibration_curves.png,
    prva verzija bez kalibracije) pokazala je sistematsku prekalibrisanost
    u višem opsegu predviđenih verovatnoća - upravo scenario najavljen u
    metodologiji ('po potrebi će biti razmotrena kalibracija ... ukoliko
    modeli pokažu dobru diskriminaciju ali lošu kalibraciju')."""
    from sklearn.calibration import CalibratedClassifierCV

    results = {}

    for label, lr_csv, xgb_csv in [("model_a", "model_a_lr.csv", "model_a_xgb.csv"),
                                     ("model_b", "model_b_lr.csv", "model_b_xgb.csv")]:
        Xtr, Xte, ytr, yte = _load_split(lr_csv)
        lr = LogisticRegression(penalty="l2", solver="lbfgs", random_state=RANDOM_STATE,
                                 **BEST_PARAMS_LR)
        lr.fit(Xtr, ytr)
        results[f"{label}_logreg"] = (yte, lr.predict_proba(Xte)[:, 1])

        lr_base = LogisticRegression(penalty="l2", solver="lbfgs", random_state=RANDOM_STATE,
                                      **BEST_PARAMS_LR)
        lr_cal = CalibratedClassifierCV(lr_base, method="isotonic", cv=5)
        lr_cal.fit(Xtr, ytr)
        results[f"{label}_logreg_calibrated"] = (yte, lr_cal.predict_proba(Xte)[:, 1])

        Xtr, Xte, ytr, yte = _load_split(xgb_csv)
        pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
                             **BEST_PARAMS_XGB)
        xgb.fit(Xtr, ytr)
        results[f"{label}_xgboost"] = (yte, xgb.predict_proba(Xte)[:, 1])

        xgb_base = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
                                  **BEST_PARAMS_XGB)
        xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
        xgb_cal.fit(Xtr, ytr)
        results[f"{label}_xgboost_calibrated"] = (yte, xgb_cal.predict_proba(Xte)[:, 1])

    return results


def plot_calibration_curves(results: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"model_a_logreg": "#1f77b4", "model_a_xgboost": "#aec7e8",
              "model_b_logreg": "#d62728", "model_b_xgboost": "#ff9896"}
    labels = {"model_a_logreg": "Model A - logistička regresija",
              "model_a_xgboost": "Model A - XGBoost",
              "model_b_logreg": "Model B - logistička regresija",
              "model_b_xgboost": "Model B - XGBoost"}

    for key, (y_true, y_prob) in results.items():
        if key not in labels:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=labels[key], color=colors[key])

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfektna kalibracija")
    ax.set_xlabel("Predviđena verovatnoća (xG)")
    ax.set_ylabel("Stvarna frekvencija gola")
    ax.set_title("Calibration Curve: Model A vs Model B")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curves.png", dpi=150)
    plt.close()
    print("Sačuvano: calibration_curves.png")


def plot_calibration_before_after(results: dict):
    """Fokusiran prikaz: Model B XGBoost, pre i posle izotone kalibracije -
    centralni model rada, najjasnija demonstracija efekta kalibracije."""
    fig, ax = plt.subplots(figsize=(7, 7))

    y_true, y_prob_raw = results["model_b_xgboost"]
    y_true_cal, y_prob_cal = results["model_b_xgboost_calibrated"]

    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=8, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true_cal, y_prob_cal, n_bins=8, strategy="quantile")

    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", color="#ff7f0e", label="Pre kalibracije")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="s", color="#2ca02c", label="Posle izotone kalibracije")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfektna kalibracija")

    ax.set_xlabel("Predviđena verovatnoća (xG)")
    ax.set_ylabel("Stvarna frekvencija gola")
    ax.set_title("Efekat post-hoc kalibracije - Model B (XGBoost)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_before_after.png", dpi=150)
    plt.close()
    print("Sačuvano: calibration_before_after.png")


def brier_comparison_table(results: dict) -> pd.DataFrame:
    """Brier score pre/posle kalibracije za sve 4 model-varijante."""
    from sklearn.metrics import brier_score_loss

    rows = []
    for base_key in ["model_a_logreg", "model_a_xgboost", "model_b_logreg", "model_b_xgboost"]:
        y_true, y_prob_raw = results[base_key]
        _, y_prob_cal = results[f"{base_key}_calibrated"]
        rows.append({
            "model": base_key,
            "brier_pre_kalibracije": round(brier_score_loss(y_true, y_prob_raw), 4),
            "brier_posle_kalibracije": round(brier_score_loss(y_true, y_prob_cal), 4),
        })
    df = pd.DataFrame(rows)
    print("\n-- Brier score: pre vs. posle post-hoc kalibracije --")
    print(df.to_string(index=False))
    df.to_csv(Path(__file__).resolve().parent.parent / "tables" / "brier_calibration_comparison.csv",
              index=False)
    return df


def plot_roc_curves(results: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    base_only = {k: v for k, v in results.items() if "calibrated" not in k}
    for key, (y_true, y_prob) in base_only.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{key} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC krive: Model A vs Model B")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print("Sačuvano: roc_curves.png")


def plot_pr_curves(results: dict):
    base_only = {k: v for k, v in results.items() if "calibrated" not in k}
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, (y_true, y_prob) in base_only.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, label=key)
    baseline = np.mean(list(base_only.values())[0][0])
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (stopa golova={baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall krive: Model A vs Model B")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close()
    print("Sačuvano: pr_curves.png")


def plot_confusion_matrices(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    order = ["model_a_logreg", "model_a_xgboost", "model_b_logreg", "model_b_xgboost"]
    titles = {"model_a_logreg": "Model A - LogReg", "model_a_xgboost": "Model A - XGBoost",
              "model_b_logreg": "Model B - LogReg", "model_b_xgboost": "Model B - XGBoost"}

    for ax, key in zip(axes.ravel(), order):
        y_true, y_prob = results[key]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Nije gol", "Gol"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(titles[key])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("Sačuvano: confusion_matrices.png")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results = get_test_predictions()
    plot_calibration_curves(results)
    plot_calibration_before_after(results)
    brier_comparison_table(results)
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrices(results)
