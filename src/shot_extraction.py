"""
shot_extraction.py
====================
Ekstrakcija šuteva iz StatsBomb event podataka i feature engineering za
Model A (klasični event atributi) i Model B (Model A + StatsBomb 360
prostorni atributi).

Koordinatni sistem StatsBomb-a: teren 120 x 80 (yard-jedinice).
Gol (centar) se nalazi na (120, 40).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

GOAL_X, GOAL_Y = 120.0, 40.0
GOAL_POST_TOP = (120.0, 36.0)     # levi gol-direk (iz perspektive napada)
GOAL_POST_BOTTOM = (120.0, 44.0)  # desni gol-direk
GOAL_WIDTH = 8.0                  # yardi (FIFA standard ~7.32 m ≈ 8 yd)


def _angle_to_goal(x: float, y: float) -> float:
    """
    Ugao (u stepenima) pod kojim šuter 'vidi' gol-okvir (shot angle).
    Standardna xG formula: ugao između linija ka oba gol-direka.
    """
    dx = GOAL_X - x
    dy_top = GOAL_POST_TOP[1] - y
    dy_bottom = GOAL_POST_BOTTOM[1] - y

    angle_top = math.atan2(dy_top, dx)
    angle_bottom = math.atan2(dy_bottom, dx)
    angle = abs(math.degrees(angle_top - angle_bottom))
    return angle


def _distance_to_goal(x: float, y: float) -> float:
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float,
                                bx: float, by: float) -> float:
    """
    Udaljenost tačke P od duži AB (segment, ne prava linija - projekcija se
    klipuje na [0,1] duž segmenta). Koristi se za 'nearest_defender_to_shot_line':
    realnija mera od cone-based pristupa, jer direktno mepи koliko je branilac
    udaljen od PRAVE linije šuter->centar gola, bez pretpostavke o širini
    koridora.
    """
    apx, apy = px - ax, py - ay
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    closest_x, closest_y = ax + t * abx, ay + t * aby
    return math.hypot(px - closest_x, py - closest_y)


def _extract_360_features(shot_event: dict) -> dict:
    """
    Iz shot['shot']['freeze_frame'] izvlači prostorne (360) atribute:
    - broj branilaca (protivnika) u radijusu trougla šuter->gol
    - broj saigrača u istom prostoru
    - da li je golman u kadru i njegova udaljenost od linije šuta
    - "otvorenost ugla" - deo ugla ka golu koji NIJE blokiran protivnicima
    Vraća dict sa NaN vrednostima ako freeze_frame ne postoji (model A
    mora moći da radi i bez ovoga).
    """
    shot = shot_event.get("shot", {})
    ff = shot.get("freeze_frame")
    loc = shot_event.get("location")

    empty = {
        "n_defenders_in_cone": np.nan,
        "n_teammates_in_cone": np.nan,
        "n_opponents_close": np.nan,
        "goalkeeper_distance": np.nan,
        "goalkeeper_angle_offset": np.nan,
        "open_goal_angle_ratio": np.nan,
        "nearest_defender_to_shot_line": np.nan,
        "pressure_score": np.nan,
        "has_360": False,
    }

    if not ff or not loc:
        return empty

    sx, sy = loc[0], loc[1]
    shot_distance = _distance_to_goal(sx, sy)
    full_angle = _angle_to_goal(sx, sy)

    n_defenders_in_cone = 0
    n_teammates_in_cone = 0
    n_opponents_close = 0
    gk_distance = np.nan
    gk_angle_offset = np.nan
    shot_line_distances = []   # udaljenosti branilaca (ispred šutera) od linije šuter->gol
    pressure_score = 0.0       # Σ 1/(d+0.5) za protivnike unutar 10yd (Football xG v3 ideja)

    # Vektor pravac šuter -> centar gola, koristimo ga da projektujemo
    # protivnike na "konus" između šutera i gol-okvira.
    to_goal_x, to_goal_y = GOAL_X - sx, GOAL_Y - sy
    to_goal_len = math.hypot(to_goal_x, to_goal_y) or 1e-9

    # Ugao (azimut) ka levom i desnom gol-direku, gledano OD šutera -
    # potreban da bismo znali koji deo [angle_bottom, angle_top] intervala
    # je "u senci" pojedinačnog branioca.
    angle_to_top = math.atan2(GOAL_POST_TOP[1] - sy, GOAL_X - sx)
    angle_to_bottom = math.atan2(GOAL_POST_BOTTOM[1] - sy, GOAL_X - sx)
    angle_lo, angle_hi = sorted([angle_to_top, angle_to_bottom])

    PLAYER_HALF_WIDTH = 0.5  # yardi (~0.45m), gruba fizička širina igrača

    blocked_intervals = []  # lista (lo, hi) uglova zaklonjenih pojedinačnim igračima

    for p in ff:
        px, py = p["location"][0], p["location"][1]
        is_teammate = p.get("teammate", False)
        position_name = (p.get("position") or {}).get("name", "")

        dist_to_shooter = math.hypot(px - sx, py - sy)

        # Da li je tačka geometrijski "između" šutera i gola (u konusu),
        # aproksimacija: projekcija na pravac šuter->gol mora biti
        # pozitivna i kraća od distance do gola, a bočno odstojanje malo.
        proj = ((px - sx) * to_goal_x + (py - sy) * to_goal_y) / to_goal_len
        perp = abs((px - sx) * (-to_goal_y) + (py - sy) * to_goal_x) / to_goal_len

        in_cone = (0 < proj < shot_distance) and (perp < 3.0)  # 3yd koridor

        if not is_teammate:
            if position_name == "Goalkeeper":
                gk_distance = dist_to_shooter
                # bočni ugao golmana u odnosu na centralnu liniju šuta
                gk_angle_offset = math.degrees(math.atan2(perp, max(proj, 1e-6)))
            else:
                if in_cone:
                    n_defenders_in_cone += 1
                if dist_to_shooter <= 5.0:
                    n_opponents_close += 1
                if dist_to_shooter <= 10.0:
                    # Kontinuirana mera pritiska - bliži branioci doprinose
                    # više, ali bez naglog "cut-off" praga (za razliku od
                    # n_defenders_in_cone/n_opponents_close, koji su binarni
                    # brojevi). Preuzeto iz alternativne implementacije
                    # ("Football xG v3") kao dopuna, ne zamena, postojećih
                    # mera pritiska.
                    pressure_score += 1.0 / (dist_to_shooter + 0.5)
                if px >= sx:
                    # Samo branioci koji su "ispred" šutera (bliže golu) su
                    # relevantni za blokiranje linije šuta - branilac iza
                    # šutera ne može fizički da blokira udarac ka golu.
                    d_line = _distance_point_to_segment(px, py, sx, sy, GOAL_X, GOAL_Y)
                    shot_line_distances.append(d_line)
                if in_cone and dist_to_shooter > 1e-6:
                    # Realna fizička "senka": polu-ugao = arctan(širina/udaljenost),
                    # centrirana na azimut igrača gledano od šutera (ne na
                    # geometrijsku sredinu gola) - ovo je suštinska razlika u
                    # odnosu na prvu (preterujuću) verziju formule.
                    player_azimuth = math.atan2(py - sy, px - sx)
                    shadow_half = math.atan2(PLAYER_HALF_WIDTH, dist_to_shooter)
                    blocked_intervals.append(
                        (player_azimuth - shadow_half, player_azimuth + shadow_half)
                    )
        else:
            if in_cone:
                n_teammates_in_cone += 1

    open_ratio = np.nan
    if full_angle > 0:
        # Klipujemo svaki interval na [angle_lo, angle_hi] (sam okvir gola),
        # spajamo preklapajuće intervale, sabiramo pokriven ugao - ovo
        # sprečava da se isti deo gola "duplo računa" sa dva branioca koji
        # stoje jedan iza drugog (čest slučaj u zbijenoj kaznenoj zoni).
        clipped = []
        for lo, hi in blocked_intervals:
            lo_c, hi_c = max(lo, angle_lo), min(hi, angle_hi)
            if lo_c < hi_c:
                clipped.append((lo_c, hi_c))
        clipped.sort()
        merged_blocked = 0.0
        cur_lo, cur_hi = None, None
        for lo, hi in clipped:
            if cur_lo is None:
                cur_lo, cur_hi = lo, hi
            elif lo <= cur_hi:
                cur_hi = max(cur_hi, hi)
            else:
                merged_blocked += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        if cur_lo is not None:
            merged_blocked += cur_hi - cur_lo

        full_angle_rad = math.radians(full_angle)
        open_ratio = max(0.0, 1.0 - min(merged_blocked, full_angle_rad) / full_angle_rad)

    nearest_defender_to_shot_line = min(shot_line_distances) if shot_line_distances else np.nan

    return {
        "n_defenders_in_cone": n_defenders_in_cone,
        "n_teammates_in_cone": n_teammates_in_cone,
        "n_opponents_close": n_opponents_close,
        "goalkeeper_distance": gk_distance,
        "goalkeeper_angle_offset": gk_angle_offset,
        "open_goal_angle_ratio": open_ratio,
        "nearest_defender_to_shot_line": nearest_defender_to_shot_line,
        "pressure_score": pressure_score,
        "has_360": True,
    }


def _game_state(shot_event: dict, score_state: dict) -> str:
    """Vraća 'leading' / 'trailing' / 'level' za tim koji šutira, u trenutku šuta."""
    team_id = shot_event["team"]["id"]
    home_id, away_id = score_state["home_id"], score_state["away_id"]
    home_goals, away_goals = score_state["home_goals"], score_state["away_goals"]

    if team_id == home_id:
        diff = home_goals - away_goals
    else:
        diff = away_goals - home_goals

    if diff > 0:
        return "leading"
    if diff < 0:
        return "trailing"
    return "level"


def extract_shots_for_match(match_id: int, tournament_key: str, match_meta: dict) -> pd.DataFrame:
    """
    Učitava events za jedan meč, izvlači sve šuteve i pravi feature-e
    (Model A + Model B atributi u istom redu; razdvajanje na A/B se radi
    kasnije, pri treniranju, izborom kolona).
    """
    events_path = RAW_DIR / "events" / f"{match_id}.json"
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    # Sortiramo hronološki da bismo mogli da pratimo game state (tekući rezultat)
    events_sorted = sorted(events, key=lambda e: (e["period"], e["index"]))

    home_id = match_meta["home_team"]["home_team_id"]
    away_id = match_meta["away_team"]["away_team_id"]
    score_state = {"home_id": home_id, "away_id": away_id, "home_goals": 0, "away_goals": 0}

    rows = []
    for e in events_sorted:
        etype = e.get("type", {}).get("name")

        # Pratimo gol (i autogol) da ažuriramo tekući rezultat PRE eventualnog
        # šuta koji ide u dataset (game state mora reflektovati stanje TOKOM
        # akcije, ne posle nje - ažuriranje radimo posle ekstrakcije šuta iz
        # iste iteracije nije problem jer je trenutni event obrađen prvo).
        if etype == "Shot":
            shot = e.get("shot", {})
            outcome = (shot.get("outcome") or {}).get("name")
            loc = e.get("location")
            if not loc or outcome is None:
                continue  # nepotpun zapis, izbacujemo

            is_goal = int(outcome == "Goal")
            distance = _distance_to_goal(loc[0], loc[1])
            angle = _angle_to_goal(loc[0], loc[1])

            row = {
                "match_id": match_id,
                "tournament": tournament_key,
                "event_id": e["id"],
                "minute": e.get("minute"),
                "period": e.get("period"),
                "team": e["team"]["name"],
                "team_id": e["team"]["id"],
                "player": (e.get("player") or {}).get("name"),
                "x": loc[0],
                "y": loc[1],
                "distance_to_goal": distance,
                "shot_angle_deg": angle,
                "body_part": (shot.get("body_part") or {}).get("name"),
                "technique": (shot.get("technique") or {}).get("name"),
                "play_pattern": (e.get("play_pattern") or {}).get("name"),
                "shot_type": (shot.get("type") or {}).get("name"),  # Open Play/Free Kick/Penalty...
                "under_pressure": bool(e.get("under_pressure", False)),
                "first_time": bool(shot.get("first_time", False)),
                "statsbomb_xg": shot.get("statsbomb_xg"),
                "game_state": _game_state(e, score_state),
                "is_goal": is_goal,
            }
            row.update(_extract_360_features(e))
            rows.append(row)

        if etype == "Own Goal Against":
            # Autogol povećava rezultat SUPROTNOG tima od onog koji je
            # počinio autogol.
            scoring_team_id = away_id if e["team"]["id"] == home_id else home_id
            if scoring_team_id == home_id:
                score_state["home_goals"] += 1
            else:
                score_state["away_goals"] += 1
        elif etype == "Shot" and (e.get("shot", {}).get("outcome") or {}).get("name") == "Goal":
            if e["team"]["id"] == home_id:
                score_state["home_goals"] += 1
            else:
                score_state["away_goals"] += 1

    return pd.DataFrame(rows)


def build_shots_dataset(verbose: bool = True) -> pd.DataFrame:
    """Iterira kroz sva tri turnira i sastavlja jedinstven shot-level dataset."""
    from data_loader import TOURNAMENTS, get_matches

    all_dfs = []
    for tkey in TOURNAMENTS:
        matches = get_matches(tkey)
        match_lookup = {m["match_id"]: m for m in matches}
        for i, mid in enumerate(match_lookup, 1):
            df = extract_shots_for_match(mid, tkey, match_lookup[mid])
            all_dfs.append(df)
            if verbose and i % 20 == 0:
                print(f"  [{tkey}] obrađeno {i}/{len(match_lookup)} mečeva...")
        if verbose:
            print(f"[{tkey}] gotovo: {len(match_lookup)} mečeva.")

    full = pd.concat(all_dfs, ignore_index=True)
    return full


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    df = build_shots_dataset()
    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "shots_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nUkupno šuteva: {len(df)}")
    print(f"Sačuvano u: {out_path}")
