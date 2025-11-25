from pathlib import Path
from typing import Dict

import pandas as pd

from .config import (
    START_DATE,
    GAME_TYPES,
    N_EARLY_GAMES_DROP,
    PROCESSED_DIR,
)


def filter_modern_regular_season(games: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    mask = (df["game_date"] >= START_DATE) & df["season_type"].isin(GAME_TYPES)
    return df.loc[mask].reset_index(drop=True)


def drop_early_games(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.sort_values(["season_id", "team_id_home", "game_date"])
    df["home_game_number"] = df.groupby(
        ["season_id", "team_id_home"]
    ).cumcount() + 1

    df = df.sort_values(["season_id", "team_id_away", "game_date"])
    df["away_game_number"] = df.groupby(
        ["season_id", "team_id_away"]
    ).cumcount() + 1

    keep = (df["home_game_number"] > n) & (df["away_game_number"] > n)
    df = df.loc[keep].drop(columns=["home_game_number", "away_game_number"])
    return df.reset_index(drop=True)


def merge_stats(
    games: pd.DataFrame,
    other_stats: pd.DataFrame,
    game_info: pd.DataFrame,
) -> pd.DataFrame:
    df = games.merge(other_stats, on="game_id", how="left")
    if "game_date" not in game_info.columns:
        game_info["game_date"] = pd.to_datetime(game_info["game_date"])
    df = df.merge(
        game_info[["game_id", "attendance", "game_time"]],
        on="game_id",
        how="left",
    )
    return df


def add_target_and_basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_win"] = (df["wl_home"] == "W").astype(int)
    return df.reset_index(drop=True)


def make_processed_games(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    games = raw["games"]
    other_stats = raw["other_stats"]
    game_info = raw["game_info"]

    df = filter_modern_regular_season(games)
    df = drop_early_games(df, N_EARLY_GAMES_DROP)
    df = merge_stats(df, other_stats, game_info)
    df = add_target_and_basic_clean(df)
    return df


def save_processed_games(df: pd.DataFrame, filename: str = "games_processed.parquet") -> None:
    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / filename, index=False)
