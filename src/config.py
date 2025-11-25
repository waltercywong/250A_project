from pathlib import Path

# Get the project root directory (parent of src/)
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent

DATA_DIR = str(_PROJECT_ROOT / "data")
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")

# Season and date filters
START_DATE = "2003-09-01"
GAME_TYPES = ["Regular Season"]

# Early games filter
N_EARLY_GAMES_DROP = 10

# Train/Val/Test split by season
# Note: NBA season_id format is YYYYY where first digit is 2, rest is year
# E.g., 22018 = 2018-19 season, 22021 = 2021-22 season
TRAIN_END_SEASON = 22018  # Up to 2018-19 season
VAL_END_SEASON = 22021    # Up to 2021-22 season
TEST_END_SEASON = 22024   # Up to 2024-25 season

# Rolling windows for feature engineering
ROLLING_WINDOWS = [5, 10]

# Discretization bins (example; can refine after EDA)
WIN_PCT_BINS = [0.4, 0.55, 0.7]
REST_DAYS_BINS = [0, 1, 3, 6]

# Random seed
RANDOM_SEED = 42
