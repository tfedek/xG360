import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier

from football_xg.config import (
    RANDOM_STATE,
    TARGET,
    CATEGORICAL,
    MODEL_B_NUMERIC,
)


def load_modeling_data(path):
    df = pd.read_csv(path)
    # Exclude penalties (no 360 data, fixed geometry, separate analysis)
    if "shot_type" in df.columns:
        df = df[df["shot_type"] != "Penalty"].reset_index(drop=True)
    df = df.drop(columns=["shot_statsbomb_xg"], errors="ignore")

    for col in MODEL_B_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].fillna("Unknown").astype(str)

    df[TARGET] = df[TARGET].astype(int)
    df["tournament"] = df["tournament"].astype(str)

    return df


def make_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def make_models(numeric_features):
    preprocessor = make_preprocessor(numeric_features)

    logistic = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    xgboost = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logistic_grid = {
        "model__C": [0.05, 0.1, 0.5, 1, 2],
        "model__class_weight": [None, "balanced"],
    }

    xgboost_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.9],
        "model__min_child_weight": [1, 5],
        "model__scale_pos_weight": [1, 3, 7],
    }

    return {
        "logistic": (logistic, logistic_grid),
        "xgboost": (xgboost, xgboost_grid),
    }


def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.03, 0.7, 200)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }