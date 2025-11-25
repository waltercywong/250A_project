DATA_DIR = "../data"
PROCESSED_DIR = "../data/processed"

# Season and date filters
START_DATE = "2003-09-01"
GAME_TYPES = ["Regular Season"]

# Early games filter
N_EARLY_GAMES_DROP = 10

# Train/Val/Test split by season
TRAIN_END_SEASON = 2018
VAL_END_SEASON = 2021
TEST_END_SEASON = 2024

# Rolling windows for feature engineering
ROLLING_WINDOWS = [5, 10]

# Discretization bins (example; can refine after EDA)
WIN_PCT_BINS = [0.4, 0.55, 0.7]
REST_DAYS_BINS = [0, 1, 3, 6]

# Random seed
RANDOM_SEED = 42
