"""
data_loader.py
================
Preuzimanje i učitavanje StatsBomb event + 360 (freeze-frame) podataka
za tri turnira sa potpunim 360 pokrićem: SP 2022, Euro 2020, Euro 2024.

Izvor: github.com/statsbomb/open-data (besplatan, javan repozitorijum)
"""

import json
import os
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Turniri sa potpunim (100%) 360 pokrićem na nivou meča (provereno ranije)
TOURNAMENTS = {
    "WC2022":   {"competition_id": 43, "season_id": 106, "label": "Svetsko prvenstvo 2022"},
    "EURO2020": {"competition_id": 55, "season_id": 43,  "label": "Evropsko prvenstvo 2020"},
    "EURO2024": {"competition_id": 55, "season_id": 282, "label": "Evropsko prvenstvo 2024"},
}


def _fetch_json(url: str, dest: Path, retries: int = 3, pause: float = 0.3) -> dict:
    """Preuzima JSON sa dozvoljenog domena i čuva lokalno (cache)."""
    if dest.exists():
        with open(dest, "r", encoding="utf-8") as f:
            return json.load(f)

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:  # noqa: BLE001 - želimo retry na bilo koju grešku mreže
            last_err = e
            time.sleep(pause)
    raise RuntimeError(f"Neuspelo preuzimanje {url}: {last_err}")


def get_matches(tournament_key: str) -> list[dict]:
    """Vraća listu mečeva (sa metapodacima) za dati turnir."""
    t = TOURNAMENTS[tournament_key]
    url = f"{BASE_URL}/matches/{t['competition_id']}/{t['season_id']}.json"
    dest = RAW_DIR / "matches" / f"{tournament_key}.json"
    return _fetch_json(url, dest)


def get_events(match_id: int) -> list[dict]:
    """Vraća listu event-objekata za dati meč."""
    url = f"{BASE_URL}/events/{match_id}.json"
    dest = RAW_DIR / "events" / f"{match_id}.json"
    return _fetch_json(url, dest)


def get_360(match_id: int) -> list[dict] | None:
    """Vraća listu 360 freeze-frame zapisa za dati meč (None ako ne postoji)."""
    url = f"{BASE_URL}/three-sixty/{match_id}.json"
    dest = RAW_DIR / "three-sixty" / f"{match_id}.json"
    try:
        return _fetch_json(url, dest)
    except RuntimeError:
        return None


def download_tournament(tournament_key: str, verbose: bool = True) -> list[int]:
    """
    Preuzima sve mečeve, evente i 360 podatke za dati turnir.
    Vraća listu match_id-jeva koji su uspešno preuzeti.
    """
    matches = get_matches(tournament_key)
    match_ids = []
    for i, m in enumerate(matches, 1):
        mid = m["match_id"]
        has_360 = m.get("match_status_360") == "available"
        get_events(mid)
        if has_360:
            get_360(mid)
        match_ids.append(mid)
        if verbose and i % 10 == 0:
            print(f"  [{tournament_key}] preuzeto {i}/{len(matches)} mečeva...")
    if verbose:
        print(f"[{tournament_key}] Završeno: {len(match_ids)} mečeva preuzeto.")
    return match_ids


def download_all(verbose: bool = True) -> dict[str, list[int]]:
    """Preuzima podatke za sva tri turnira."""
    result = {}
    for key in TOURNAMENTS:
        if verbose:
            print(f"\n=== Preuzimanje: {TOURNAMENTS[key]['label']} ({key}) ===")
        result[key] = download_tournament(key, verbose=verbose)
    return result


if __name__ == "__main__":
    ids = download_all()
    for k, v in ids.items():
        print(f"{k}: {len(v)} mečeva")
