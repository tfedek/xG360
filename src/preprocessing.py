"""
preprocessing.py
==================
Korak 1 (EDA) i korak 3 (priprema/pretprocesiranje) metodologije:

- Učitava sirovi shot dataset (shots_raw.csv)
- Izdvaja penale u posebnu deskriptivnu tabelu (ne ulaze u Model A/B -
  odluka eksplicitno potvrđena: penali nemaju 360 podatke kod StatsBomb-a
  i kvalitativno su drugačiji tip šuta - fiksna udaljenost, duel 1v1)
- Sprovodi EDA: distribucije, missing values, outlieri, korelacije
- Kodira kategorijske promenljive
- Vraća čist DataFrame spreman za feature engineering / modelovanje

Pokretanje: python3 preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"

# Kolone koje čine Model A (klasični event atributi, bez 360)
MODEL_A_NUMERIC = [
    "distance_to_goal", "shot_angle_deg", "minute",
]
MODEL_A_CATEGORICAL = [
    "body_part_clean", "shot_type_clean", "play_pattern", "game_state", "technique",
]
MODEL_A_BINARY = ["under_pressure", "first_time"]

# Dodatne kolone koje Model B dodaje na Model A (StatsBomb 360 prostorni atributi)
MODEL_B_EXTRA_NUMERIC = [
    "n_defenders_in_cone", "n_teammates_in_cone", "n_opponents_close",
    "goalkeeper_distance", "open_goal_angle_ratio",
    "nearest_defender_to_shot_line", "pressure_score",
]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "shots_raw.csv")
    return df


def split_penalties(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Razdvaja penale (deskriptivna kategorija) od open-play/free-kick/corner
    šuteva (idu u Model A/B). Odluka potvrđena u toku konsultacije: penali
    su kvalitativno drugačiji (fiksna udaljenost, nema 360 podataka).
    """
    is_pen = df["shot_type"] == "Penalty"
    penalties = df[is_pen].copy()
    modeling = df[~is_pen].copy()
    return modeling, penalties


def clean_body_part(df: pd.DataFrame) -> pd.DataFrame:
    """Grupiše 'Left Foot'/'Right Foot' -> 'Foot' (lateralnost noge nije
    relevantna za xG samu po sebi; razdvajanje bi samo udvostručilo broj
    nivoa kategorije bez dodatne prediktivne vrednosti za ovo pitanje)."""
    df = df.copy()
    mapping = {
        "Left Foot": "Foot", "Right Foot": "Foot",
        "Head": "Head", "Other": "Other",
    }
    df["body_part_clean"] = df["body_part"].map(mapping).fillna("Other")
    return df


def clean_shot_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregira retke shot_type kategorije (Corner: 6 opservacija/0 golova,
    Free Kick: 130 opservacija/3 gola - 2.3%) u jedinstvenu 'Set Piece'
    kategoriju. Razlog: obe kategorije imaju toliko malo opservacija i/ili
    toliko nisku stopu gola da pojedinačno uzrokuju kvazi-savršenu
    separaciju u logističkoj regresiji (Corner: SVI ishodi su 0 - potpuna
    separacija). Agregacija na izvoru je metodološki čistije rešenje od
    post-hoc regularizacije, i ne gubi suštinsku informaciju - oba tipa
    šuta dele zajedničku osobinu (indirektna situacija iz prekida, ne iz
    igre u toku)."""
    df = df.copy()
    df["shot_type_clean"] = df["shot_type"].replace(
        {"Corner": "Set Piece", "Free Kick": "Set Piece"}
    )
    return df


def eda_report(df: pd.DataFrame, label: str = "modeling set") -> dict:
    """
    Sprovodi EDA korak: distribucije, missing values, outlieri (IQR metoda),
    korelacije numeričkih atributa. Vraća dict sa rezultatima (i ispisuje
    sažetak na ekran) - rezultati se kasnije koriste za tabele u radu.
    """
    report = {}
    print(f"\n{'='*60}\nEDA: {label} (n={len(df)})\n{'='*60}")

    # --- Distribucije numeričkih promenljivih ---
    numeric_cols = ["distance_to_goal", "shot_angle_deg", "minute"] + MODEL_B_EXTRA_NUMERIC
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    desc = df[numeric_cols].describe().T
    report["descriptive_stats"] = desc
    print("\n-- Deskriptivna statistika numeričkih promenljivih --")
    print(desc)

    # --- Missing values ---
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    report["missing_values"] = missing
    print("\n-- Nedostajuće vrednosti (kolone sa >0) --")
    print(missing if len(missing) else "Nema nedostajućih vrednosti.")

    # --- Outlieri (IQR metoda, 1.5x pravilo) ---
    outlier_summary = {}
    for col in ["distance_to_goal", "shot_angle_deg"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_summary[col] = {"n_outliers": int(n_out), "pct": round(100 * n_out / len(df), 2),
                                 "lower_bound": round(lo, 2), "upper_bound": round(hi, 2)}
    report["outliers"] = outlier_summary
    print("\n-- Outlieri (IQR 1.5x metoda) --")
    for col, info in outlier_summary.items():
        print(f"  {col}: {info['n_outliers']} ({info['pct']}%), granice=[{info['lower_bound']}, {info['upper_bound']}]")

    # --- Korelacije numeričkih atributa ---
    corr = df[numeric_cols].corr(numeric_only=True)
    report["correlation_matrix"] = corr
    print("\n-- Korelacije numeričkih atributa (Pearson) --")
    print(corr.round(2))

    # --- Class balance ---
    goal_rate = df["is_goal"].mean()
    report["class_balance"] = {"n_goals": int(df["is_goal"].sum()), "n_total": len(df),
                                "goal_rate": round(goal_rate, 4), "ratio": round((1 - goal_rate) / goal_rate, 2)}
    print(f"\n-- Class balance -- golova: {df['is_goal'].sum()}/{len(df)} ({100*goal_rate:.1f}%), "
          f"odnos ne-gol:gol ≈ {(1-goal_rate)/goal_rate:.1f}:1")

    return report


def penalty_descriptive_summary(penalties: pd.DataFrame) -> pd.DataFrame:
    """Deskriptivna tabela za penale (van glavnog modela)."""
    summary = pd.DataFrame({
        "n_penalties": [len(penalties)],
        "n_scored": [int(penalties["is_goal"].sum())],
        "conversion_rate": [round(penalties["is_goal"].mean(), 4)],
    })
    return summary


def build_model_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Od očišćenog modeling seta pravi finalne tabele za Model A i Model B,
    sa one-hot kodiranjem kategorijskih promenljivih. Šutevi kojima
    nedostaju 360 atributi (has_360 == False, a NISU penali - npr. retki
    edge-case eventi) izbacuju se SAMO iz Model B, ne iz Model A, da bi
    Model A iskoristio maksimalan broj opservacija.

    Posebna napomena o 'nearest_defender_to_shot_line': NaN za ovaj feature
    ne znači nedostajući podatak u uobičajenom smislu - javlja se kada NEMA
    branilaca između šutera i gola (svi protivnici su 'iza' šutera), što je
    faktički najbolji mogući scenario za šansu da se postigne gol (čista
    linija pucanja). Popunjava se sentinel vrednošću većom od bilo koje
    stvarno opažene udaljenosti (umesto medijane, koja bi pogrešno opisala
    ovu situaciju kao 'prosečnu', ili listwise izbacivanja, koje bi
    nepotrebno smanjilo uzorak za ~9%).
    """
    df = df.copy()

    NEAREST_DEFENDER_SENTINEL = df["nearest_defender_to_shot_line"].max() * 1.5 \
        if df["nearest_defender_to_shot_line"].notna().any() else 30.0
    df["nearest_defender_to_shot_line"] = df["nearest_defender_to_shot_line"].fillna(
        NEAREST_DEFENDER_SENTINEL
    )

    # Binarne kolone -> int
    for col in MODEL_A_BINARY:
        df[col] = df[col].astype(int)

    cat_dummies = pd.get_dummies(
        df[MODEL_A_CATEGORICAL], prefix=MODEL_A_CATEGORICAL, drop_first=True
    )

    base_cols = ["match_id", "tournament", "team", "player", "is_goal", "statsbomb_xg"]
    model_a = pd.concat(
        [df[base_cols + MODEL_A_NUMERIC + MODEL_A_BINARY].reset_index(drop=True),
         cat_dummies.reset_index(drop=True)],
        axis=1,
    )

    # Model B = Model A + 360 atributi, samo za redove gde 360 postoji
    has_360_mask = df["has_360"].fillna(False).astype(bool)
    model_b = pd.concat(
        [df.loc[has_360_mask, base_cols + MODEL_A_NUMERIC + MODEL_A_BINARY].reset_index(drop=True),
         cat_dummies.loc[has_360_mask].reset_index(drop=True),
         df.loc[has_360_mask, MODEL_B_EXTRA_NUMERIC].reset_index(drop=True)],
        axis=1,
    )

    return {"model_a": model_a, "model_b": model_b}


def run_pipeline(verbose: bool = True) -> dict:
    df_raw = load_raw()
    modeling, penalties = split_penalties(df_raw)
    modeling = clean_body_part(modeling)
    modeling = clean_shot_type(modeling)

    if verbose:
        print(f"Ukupno šuteva (sirovo): {len(df_raw)}")
        print(f"Penali izdvojeni (deskriptivno): {len(penalties)}")
        print(f"Šuteva za modelovanje (open play / free kick / corner): {len(modeling)}")

    eda_main = eda_report(modeling, label="Modeling set (bez penala)")
    pen_summary = penalty_descriptive_summary(penalties)

    if verbose:
        print("\n-- Deskriptivni sažetak penala --")
        print(pen_summary)

    frames = build_model_frames(modeling)

    if verbose:
        print(f"\nModel A (klasični atributi): {frames['model_a'].shape[0]} redova, "
              f"{frames['model_a'].shape[1]} kolona")
        print(f"Model B (klasični + 360 atributi): {frames['model_b'].shape[0]} redova, "
              f"{frames['model_b'].shape[1]} kolona")
        goal_rate_a = frames["model_a"]["is_goal"].mean()
        goal_rate_b = frames["model_b"]["is_goal"].mean()
        print(f"Stopa golova Model A: {100*goal_rate_a:.2f}% | Model B: {100*goal_rate_b:.2f}%")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    eda_main["descriptive_stats"].to_csv(TABLES_DIR / "eda_descriptive_stats.csv")
    eda_main["correlation_matrix"].to_csv(TABLES_DIR / "eda_correlation_matrix.csv")
    pen_summary.to_csv(TABLES_DIR / "penalty_descriptive_summary.csv", index=False)

    frames["model_a"].to_csv(PROCESSED_DIR / "model_a.csv", index=False)
    frames["model_b"].to_csv(PROCESSED_DIR / "model_b.csv", index=False)
    penalties.to_csv(PROCESSED_DIR / "penalties.csv", index=False)

    return {"modeling": modeling, "penalties": penalties, "eda": eda_main,
            "model_a": frames["model_a"], "model_b": frames["model_b"]}


if __name__ == "__main__":
    run_pipeline()
