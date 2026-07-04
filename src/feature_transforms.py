"""
feature_transforms.py
========================
Rešava dva nalaza iz koraka provere pretpostavki (assumption_checks.py):

  1. Multikolinearnost: goalkeeper_distance i distance_to_goal (VIF 25-30,
     r=0.97) - golman po pravilu stoji blizu sopstvenog gola, pa je njegova
     udaljenost od šutera skoro identična distance_to_goal. Rešenje:
     goalkeeper_distance se zamenjuje sa "goalkeeper_anomaly" - rezidualom
     iz proste linearne regresije goalkeeper_distance ~ distance_to_goal.
     Pozitivan rezidual = golman je DALJE nego što bi se očekivalo na toj
     udaljenosti šuta (npr. izašao prerano, ili šut iz ugla); negativan =
     golman je BLIŽE nego očekivano (npr. dobro pozicioniran, presekao ugao).

  2. Linearnost logita narušena za distance_to_goal i shot_angle_deg
     (Box-Tidwell p<0.001 za oba). Rešenje SAMO za logističku regresiju:
     dodaju se log-transformisane verzije kao dodatni/zamenski prediktori.
     XGBoost ne zahteva ovu transformaciju (stabla prirodno modeluju
     nelinearne odnose), pa se primenjuje samo u model varijanti za LR.

Pokretanje: python3 feature_transforms.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def add_goalkeeper_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zamenjuje goalkeeper_distance sa goalkeeper_anomaly (rezidual nakon
    uklanjanja linearne zavisnosti od distance_to_goal). Samo za redove
    gde goalkeeper_distance postoji (Model B podskup); ako kolona ne
    postoji u datom DataFrame-u (npr. Model A), ništa se ne radi.
    """
    df = df.copy()
    if "goalkeeper_distance" not in df.columns:
        return df

    mask = df["goalkeeper_distance"].notna() & df["distance_to_goal"].notna()
    x = df.loc[mask, "distance_to_goal"].to_numpy()
    y = df.loc[mask, "goalkeeper_distance"].to_numpy()

    # Prosta linearna regresija y = a*x + b (manuelno, da ne uvodimo
    # zavisnost od dodatne biblioteke za jedan red računa)
    a, b = np.polyfit(x, y, deg=1)
    predicted = a * x + b
    residual = y - predicted

    df["goalkeeper_anomaly"] = np.nan
    df.loc[mask, "goalkeeper_anomaly"] = residual
    df = df.drop(columns=["goalkeeper_distance"])
    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje log-transformisane verzije distance_to_goal i shot_angle_deg
    (log1p da se izbegne log(0)) - koriste se SAMO u logistic-regression
    varijanti feature seta, kao zamena za sirove vrednosti, da se
    adresira narušena linearnost logita utvrđena Box-Tidwell testom.
    """
    df = df.copy()
    df["log_distance_to_goal"] = np.log1p(df["distance_to_goal"])
    df["log_shot_angle_deg"] = np.log1p(df["shot_angle_deg"])
    return df


def build_lr_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verzija feature seta namenjena logističkoj regresiji: sirovi
    distance_to_goal / shot_angle_deg se ZAMENJUJU log-transformisanim
    verzijama (ne dodaju se kao ekstra kolone - to bi samo unelo novu
    multikolinearnost, jer su log(x) i x i dalje jako korelisani).
    """
    df = add_log_transforms(df)
    df = df.drop(columns=["distance_to_goal", "shot_angle_deg"])
    df = df.rename(columns={
        "log_distance_to_goal": "distance_to_goal_log",
        "log_shot_angle_deg": "shot_angle_deg_log",
    })
    return df


def process_all() -> dict[str, pd.DataFrame]:
    model_a = pd.read_csv(PROCESSED_DIR / "model_a.csv")
    model_b = pd.read_csv(PROCESSED_DIR / "model_b.csv")

    # Korak 1: rešavanje VIF problema (samo Model B ima goalkeeper_distance)
    model_b = add_goalkeeper_anomaly(model_b)

    # Korak 2: XGBoost varijante koriste sirove (netransformisane) udaljenost/ugao
    model_a_xgb = model_a.copy()
    model_b_xgb = model_b.copy()

    # Korak 3: LR varijante koriste log-transformisane verzije
    model_a_lr = build_lr_variant(model_a)
    model_b_lr = build_lr_variant(model_b)

    out = {
        "model_a_xgb": model_a_xgb, "model_b_xgb": model_b_xgb,
        "model_a_lr": model_a_lr, "model_b_lr": model_b_lr,
    }
    for name, frame in out.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    frames = process_all()
    for name, df in frames.items():
        print(f"{name}: {df.shape[0]} redova, {df.shape[1]} kolona")
        if "goalkeeper_anomaly" in df.columns:
            print(f"  goalkeeper_anomaly: mean={df['goalkeeper_anomaly'].mean():.3f}, "
                  f"std={df['goalkeeper_anomaly'].std():.3f}")
