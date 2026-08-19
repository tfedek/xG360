from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # xg_project_v2/

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
MODELS_DIR = BASE_DIR / "data" / "models"

MATCHES_CSV = RAW_DIR / "all_matches_selected.csv"
EVENTS_DIR = RAW_DIR / "events"
THREE_SIXTY_DIR = RAW_DIR / "three-sixty"

DATASET_PATH = PROCESSED_DIR / "shots_with_360_dataset.csv"

RANDOM_STATE = 42
TARGET = "goal"

GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0
LEFT_POST_Y = GOAL_Y - GOAL_WIDTH / 2
RIGHT_POST_Y = GOAL_Y + GOAL_WIDTH / 2

BOX_X_MIN = 102.0
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0


# ============================================================
# CLEAN FEATURE SETS
# ============================================================
# Deliberately excluded:
# - shot_statsbomb_xg: leakage
# - minute/period: can introduce unstable/contextual artefacts
# - shot_technique: unstable with small category counts
# - shot_open_goal: rare/extreme, can inflate estimates
# - visible_goal_ratio/open_angle_ratio/corridor area: too noisy/redundant in current implementation


MODEL_A_NUMERIC = [
    "distance",
    "angle",
    "shot_first_time",
    "shot_one_on_one",
]


MODEL_B_NUMERIC = MODEL_A_NUMERIC + [
    # goalkeeper
    "goalkeeper_distance_360",

    # direct pressure
    "nearest_defender_distance_360",
    "defenders_within_5m_360",
    "defenders_within_10m_360",

    # shooting lane / crowding
    "opponents_between_shooter_and_goal_360",
    "opponents_in_box_360",

    # NEW: defenders close to actual shot line toward centre of goal
    "nearest_defender_to_shot_line_360",
    "defenders_within_1m_of_shot_line_360",
    "defenders_within_2m_of_shot_line_360",

    # pressure summary
    "pressure_score_360",

    # open goal angle (proportion of goal visible/unblocked)
    "open_angle_ratio_360",
]


CATEGORICAL = [
    "shot_body_part",
    "shot_type",
]
