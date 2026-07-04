"""
export_for_web.py
====================
Pravi sve JSON fajlove potrebne za web aplikaciju:

  web/data/shots.json        - svi šutevi sa feature-ima i sirovim
                                koordinatama (za listu/pretragu/teren)
  web/data/freeze_frames.json - freeze-frame pozicije igrača po šutu
                                (key = event_id)
  web/data/model_coefs.json  - koeficijenti logističke regresije
                                (Model A i Model B) - JS računa sigmoid
                                uživo, model se NE poziva na serveru

Pokretanje: python3 export_for_web.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"
WEB_DATA_DIR = PROJECT_DIR / "web" / "data"
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

TOURNAMENT_LABELS = {
    "WC2022": "Svetsko prvenstvo 2022",
    "EURO2020": "Evropsko prvenstvo 2020",
    "EURO2024": "Evropsko prvenstvo 2024",
}


def export_model_coefficients():
    """Izvozi koeficijente Model A i Model B logističke regresije kao
    JSON - JS računa sigmoid(intercept + sum(coef_i * x_i)) direktno,
    bez poziva serveru. Takođe izvozi koeficijente goalkeeper_anomaly
    regresije (a, b) jer je potrebna da se izračuna taj feature uživo."""
    bundle_a = joblib.load(PROCESSED_DIR / "fitted_models" / "model_a_logreg.joblib")
    bundle_b = joblib.load(PROCESSED_DIR / "fitted_models" / "model_b_logreg.joblib")

    shots_raw = pd.read_csv(PROCESSED_DIR / "shots_raw.csv")
    modeling = shots_raw[shots_raw["shot_type"] != "Penalty"]
    mask = modeling["goalkeeper_distance"].notna() & modeling["distance_to_goal"].notna()
    gk_a, gk_b = np.polyfit(
        modeling.loc[mask, "distance_to_goal"].to_numpy(),
        modeling.loc[mask, "goalkeeper_distance"].to_numpy(),
        deg=1,
    )

    out = {
        "model_a": {
            "intercept": float(bundle_a["model"].intercept_[0]),
            "features": bundle_a["feature_names"],
            "coefficients": bundle_a["model"].coef_[0].tolist(),
        },
        "model_b": {
            "intercept": float(bundle_b["model"].intercept_[0]),
            "features": bundle_b["feature_names"],
            "coefficients": bundle_b["model"].coef_[0].tolist(),
        },
        "goalkeeper_anomaly_regression": {"a": float(gk_a), "b": float(gk_b)},
    }
    with open(WEB_DATA_DIR / "model_coefs.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Sačuvano: model_coefs.json ({len(out['model_a']['features'])} + "
          f"{len(out['model_b']['features'])} feature-a)")


def export_shap_importance():
    """Izvozi SHAP važnost (Model B) za prikaz na 'SHAP' tabu sajta."""
    imp = pd.read_csv(PROJECT_DIR / "tables" / "shap_importance_comparison_A_vs_B.csv")
    out = imp.to_dict(orient="records")
    with open(WEB_DATA_DIR / "shap_importance.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Sačuvano: shap_importance.json ({len(out)} feature-a)")


def export_evaluation_summary():
    """Izvozi K-Fold i LOTO sažetke za prikaz performansi modela na sajtu."""
    kfold = pd.read_csv(PROJECT_DIR / "tables" / "kfold_summary.csv", index_col=0)
    loto = pd.read_csv(PROJECT_DIR / "tables" / "loto_summary.csv", index_col=0)
    brier = pd.read_csv(PROJECT_DIR / "tables" / "brier_calibration_comparison.csv")

    out = {
        "kfold": kfold.reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
        "loto": loto.reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
        "brier_calibration": brier.to_dict(orient="records"),
    }
    with open(WEB_DATA_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("Sačuvano: evaluation_summary.json")


def export_shots_and_frames():
    """
    Glavni export: za svaki šut (open-play/free-kick/corner BEZ filtriranja,
    PLUS penal-golove iz REGULARNE igre - ne promašene penale, da se izbegne
    preplavljivanje liste nebitnim promašajima) izvozi sirove koordinate,
    sve feature-e potrebne za oba modela, i freeze-frame pozicije igrača.

    KRITIČNO: penal-shootout (period == 5, StatsBomb-ova oznaka za penale
    nakon produžetaka) se EKSPLICITNO izbacuje iz CELE liste, ne samo iz
    Model A/B. Shootout nije deo same utakmice u xG smislu (igra se nakon
    što je meč zvanično završen, drugačiji psihološki/takmičarski kontekst)
    - prikazivanje shootout penala kao "gol igrača X" bilo bi netačno bez
    obzira na to da li im se daje Model A/B procena ili ne. Ranija verzija
    ovog koda je filtrirala samo po shot_type=='Penalty', što je propustilo
    da razdvoji in-game penale (legitimni deo utakmice) od shootout penala
    (Bellingham 121', Messi, Mbappé itd. - 103 šuta kroz sva tri turnira).

    Provereno: shootout šutevi NIKADA nisu ušli u Model A/B trening skup
    (svi imaju shot_type=='Penalty', pa ih je build_model_frames već
    filtrirao pre treniranja) - ispravka ovde utiče SAMO na web prikaz,
    ne na rezultate analize/rada.
    """
    shots_raw = pd.read_csv(PROCESSED_DIR / "shots_raw.csv")
    shots_raw = shots_raw[shots_raw["period"] != 5].copy()  # izbaci shootout odmah, na izvoru

    is_penalty = shots_raw["shot_type"] == "Penalty"
    modeling = shots_raw[~is_penalty].copy()
    penalty_goals = shots_raw[is_penalty & (shots_raw["is_goal"] == 1)].copy()
    combined = pd.concat([modeling, penalty_goals], ignore_index=True)

    shots_out = []
    frames_out = {}

    for tkey in ["WC2022", "EURO2020", "EURO2024"]:
        sub = combined[combined["tournament"] == tkey]
        match_ids = sub["match_id"].unique().tolist()

        for mid in match_ids:
            events_path = RAW_DIR / "events" / f"{int(mid)}.json"
            with open(events_path, "r", encoding="utf-8") as f:
                events = json.load(f)
            events_by_id = {e["id"]: e for e in events}

            match_rows = sub[sub["match_id"] == mid]
            for _, row in match_rows.iterrows():
                event_id = row["event_id"]
                ev = events_by_id.get(event_id)
                if ev is None:
                    continue

                shot = ev.get("shot", {})
                ff = shot.get("freeze_frame")

                shots_out.append({
                    "event_id": event_id,
                    "match_id": int(mid),
                    "tournament": tkey,
                    "tournament_label": TOURNAMENT_LABELS[tkey],
                    "team": row["team"],
                    "player": row["player"],
                    "minute": int(row["minute"]) if pd.notna(row["minute"]) else None,
                    "period": int(row["period"]) if pd.notna(row["period"]) else None,
                    "x": row["x"], "y": row["y"],
                    "distance_to_goal": row["distance_to_goal"],
                    "shot_angle_deg": row["shot_angle_deg"],
                    "body_part": row["body_part"],
                    "technique": row["technique"],
                    "play_pattern": row["play_pattern"],
                    "shot_type": row["shot_type"],
                    "is_penalty": bool(row["shot_type"] == "Penalty"),
                    "under_pressure": bool(row["under_pressure"]),
                    "first_time": bool(row["first_time"]),
                    "game_state": row["game_state"],
                    "is_goal": int(row["is_goal"]),
                    "statsbomb_xg": row["statsbomb_xg"] if pd.notna(row["statsbomb_xg"]) else None,
                    "n_defenders_in_cone": row["n_defenders_in_cone"] if pd.notna(row["n_defenders_in_cone"]) else None,
                    "n_teammates_in_cone": row["n_teammates_in_cone"] if pd.notna(row["n_teammates_in_cone"]) else None,
                    "n_opponents_close": row["n_opponents_close"] if pd.notna(row["n_opponents_close"]) else None,
                    "goalkeeper_distance": row["goalkeeper_distance"] if pd.notna(row["goalkeeper_distance"]) else None,
                    "open_goal_angle_ratio": row["open_goal_angle_ratio"] if pd.notna(row["open_goal_angle_ratio"]) else None,
                    "nearest_defender_to_shot_line": row["nearest_defender_to_shot_line"] if pd.notna(row["nearest_defender_to_shot_line"]) else None,
                    "pressure_score": row["pressure_score"] if pd.notna(row["pressure_score"]) else None,
                    "has_360": bool(row["has_360"]) if pd.notna(row["has_360"]) else False,
                })

                if ff:
                    frames_out[event_id] = {
                        "shooter": {"x": row["x"], "y": row["y"]},
                        "players": [
                            {
                                "x": p["location"][0], "y": p["location"][1],
                                "teammate": p.get("teammate", False),
                                "position": (p.get("position") or {}).get("name", ""),
                            }
                            for p in ff
                        ],
                    }

    with open(WEB_DATA_DIR / "shots.json", "w", encoding="utf-8") as f:
        json.dump(shots_out, f, ensure_ascii=False)
    with open(WEB_DATA_DIR / "freeze_frames.json", "w", encoding="utf-8") as f:
        json.dump(frames_out, f, ensure_ascii=False)

    print(f"Sačuvano: shots.json ({len(shots_out)} šuteva)")
    print(f"Sačuvano: freeze_frames.json ({len(frames_out)} sa 360 podacima)")


if __name__ == "__main__":
    export_model_coefficients()
    export_shap_importance()
    export_evaluation_summary()
    export_shots_and_frames()
