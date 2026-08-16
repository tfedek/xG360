import json
from pathlib import Path
import numpy as np

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def name(v):
    return v.get("name") if isinstance(v, dict) else None

def safe_loc(event):
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
