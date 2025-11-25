from pathlib import Path
import pandas as pd

from .config import DATA_DIR

DATA_PATH = Path(DATA_DIR)


def load_games() -> pd.DataFrame:
    path = DATA_PATH / "game.csv"
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_other_stats() -> pd.DataFrame:
    path = DATA_PATH / "other_stats.csv"
    return pd.read_csv(path)


def load_inactive_players() -> pd.DataFrame:
    path = DATA_PATH / "inactive_players.csv"
    return pd.read_csv(path)


def load_game_info() -> pd.DataFrame:
    path = DATA_PATH / "game_info.csv"
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {
        "games": load_games(),
        "other_stats": load_other_stats(),
        "inactive_players": load_inactive_players(),
        "game_info": load_game_info(),
    }
