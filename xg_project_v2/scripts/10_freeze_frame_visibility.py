"""
10_freeze_frame_visibility.py  -  NEW (professor review, visibility limitation)
=================================================================================
Analyzes the completeness of StatsBomb 360 freeze-frame data:
  - Number of visible players per freeze-frame
  - Number of visible opponents per freeze-frame
  - Median, IQR, range of these values
  - Differences between tournaments
  - Sensitivity check: does Model B advantage hold on shots with
    >= N visible opponents (N=5, N=8 as pre-defined thresholds)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from football_xg.config import (
    DATASET_PATH, OUTPUT_DIR, TARGET,
    MODEL_A_NUMERIC, MODEL_B_NUMERIC, CATEGORICAL,
)
from football_xg.modeling import load_modeling_data
from football_xg.data_utils import ensure_dirs

OUT_DIR = OUTPUT_DIR / "freeze_frame_visibility"
ensure_dirs(OUT_DIR)


def load_visibility_data():
    df = load_modeling_data(DATASET_PATH)
    df = df[df["shot_type"] != "Penalty"].copy().reset_index(drop=True)

    vis_cols = ["n_visible_players", "n_visible_opponents"]
    missing = [c for c in vis_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] Kolone nedostaju: {missing}")
        print("  Dodaj ove kolone u 02_build_dataset.py iz StatsBomb freeze_frame JSON-a.")
        print("  n_visible_players = len(freeze_frame)")
        print("  n_visible_opponents = sum(1 for p in freeze_frame if not p['teammate'])")
        return None
    return df


def descriptive_stats(df):
    print("\n=== Deskriptivna statistika vidljivosti freeze-frame-a ===")

    for col in ["n_visible_players", "n_visible_opponents"]:
        print(f"\n{col}:")
        print(f"  Ukupno: median={df[col].median():.1f}, "
              f"IQR=[{df[col].quantile(0.25):.1f}, {df[col].quantile(0.75):.1f}], "
              f"raspon=[{df[col].min()}, {df[col].max()}]")

        by_tournament = df.groupby("tournament")[col].agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        print(by_tournament.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["n_visible_players", "n_visible_opponents"]):
        for tournament in df["tournament"].unique():
            subset = df[df["tournament"] == tournament][col]
            ax.hist(subset, bins=range(0, 25), alpha=0.6, label=tournament)
        ax.set_xlabel(col)
        ax.set_ylabel("Broj suteva")
        ax.legend(fontsize=7)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "visibility_distribution.png", dpi=160)
    plt.close()
    print(f"\nSaved: {OUT_DIR / 'visibility_distribution.png'}")


def make_pipe(numeric_features):
    prep = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(C=0.5, max_iter=4000, random_state=42))
    ])


def loto_auc(df, numeric_features):
    data = df.copy()
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(
            data[col].median())
    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str)

    X = data[numeric_features + CATEGORICAL]
    y = data[TARGET].astype(int)
    groups = data["tournament"]

    aucs = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pipe = make_pipe(numeric_features)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], pipe.predict_proba(X.iloc[te])[:, 1]))
    return np.mean(aucs)


def sensitivity_analysis(df):
    if "n_visible_opponents" not in df.columns:
        return

    print("\n=== Sensitivity analiza: da li Model B prednost vazi za razlicite pragove vidljivosti? ===")

    thresholds = [
        ("Svi sutevi (bez praga)", None),
        (">=5 vidljivih protivnika", 5),
        (">=8 vidljivih protivnika", 8),
    ]

    rows = []
    for label, min_opp in thresholds:
        if min_opp is not None:
            subset = df[df["n_visible_opponents"] >= min_opp].copy()
        else:
            subset = df.copy()

        if len(subset) < 100:
            print(f"  {label}: premalo podataka ({len(subset)} suteva), preskoceno.")
            continue

        auc_a = loto_auc(subset, MODEL_A_NUMERIC)
        auc_b = loto_auc(subset, MODEL_B_NUMERIC)

        row = {
            "subset": label, "n_shots": len(subset),
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "delta": round(auc_b - auc_a, 4),
        }
        rows.append(row)
        print(f"  {label}: n={len(subset)}, AUC_A={auc_a:.4f}, "
              f"AUC_B={auc_b:.4f}, delta={auc_b-auc_a:+.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "visibility_sensitivity.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'visibility_sensitivity.csv'}")


def main():
    df = load_visibility_data()
    if df is None:
        print("\nNije moguce sprovesti analizu bez visibility kolona.")
        print("Dodaj sledeci kod u 02_build_dataset.py pri obradi freeze_frame-a:")
        print("""
  # U funkciji koja procesira freeze_frame po sutu:
  n_visible = len(freeze_frame) if freeze_frame else 0
  n_visible_opp = sum(1 for p in freeze_frame
                      if not p.get('teammate', True)) if freeze_frame else 0
  shot_row['n_visible_players'] = n_visible
  shot_row['n_visible_opponents'] = n_visible_opp
        """)
        return

    descriptive_stats(df)
    sensitivity_analysis(df)
    print(f"\nSvi outputi: {OUT_DIR}")


if __name__ == "__main__":
    main()
