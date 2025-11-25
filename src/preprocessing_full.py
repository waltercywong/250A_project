"""
Comprehensive preprocessing pipeline for NBA game outcome prediction.
Implements all features described in data_processing.md.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import (
    START_DATE,
    GAME_TYPES,
    N_EARLY_GAMES_DROP,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
)


# =============================================================================
# SECTION 1: BASIC FILTERING AND CLEANING
# =============================================================================

def filter_modern_regular_season(games: pd.DataFrame) -> pd.DataFrame:
    """
    Filter games to modern era (post START_DATE) and regular season only.
    
    Args:
        games: Raw games dataframe from game.csv
        
    Returns:
        Filtered dataframe with regular season games from modern era
    """
    df = games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    mask = (df["game_date"] >= START_DATE) & df["season_type"].isin(GAME_TYPES)
    return df.loc[mask].reset_index(drop=True)


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target variable: 1 = home win, 0 = away win."""
    df = df.copy()
    df["game_outcome"] = (df["wl_home"] == "W").astype(int)
    return df


# =============================================================================
# SECTION 2: ROLLING STATISTICS (Season-to-date averages)
# =============================================================================

def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics for each team based on all prior games in the season.
    This uses an expanding window to calculate season-to-date averages.
    
    Features computed (per team):
    - Win percentage
    - Points per game (PPG)
    - Opponent points per game (defensive metric)
    - Field goal percentage
    - 3-point percentage
    - Free throw percentage
    - Rebounds per game
    - Assists per game
    - Turnovers per game
    - Steals + Blocks per game
    """
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    
    # We'll create a long-format dataframe where each game appears twice (once per team)
    home_games = df[[
        "game_id", "game_date", "season_id", "team_id_home", "wl_home",
        "pts_home", "pts_away", "fg_pct_home", "fg3_pct_home", "ft_pct_home",
        "reb_home", "ast_home", "tov_home", "stl_home", "blk_home"
    ]].copy()
    home_games.columns = [
        "game_id", "game_date", "season_id", "team_id", "wl",
        "pts", "opp_pts", "fg_pct", "fg3_pct", "ft_pct",
        "reb", "ast", "tov", "stl", "blk"
    ]
    home_games["is_home"] = 1
    
    away_games = df[[
        "game_id", "game_date", "season_id", "team_id_away", "wl_away",
        "pts_away", "pts_home", "fg_pct_away", "fg3_pct_away", "ft_pct_away",
        "reb_away", "ast_away", "tov_away", "stl_away", "blk_away"
    ]].copy()
    away_games.columns = [
        "game_id", "game_date", "season_id", "team_id", "wl",
        "pts", "opp_pts", "fg_pct", "fg3_pct", "ft_pct",
        "reb", "ast", "tov", "stl", "blk"
    ]
    away_games["is_home"] = 0
    
    # Combine into long format
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(["season_id", "team_id", "game_date"]).reset_index(drop=True)
    
    # Create win indicator
    all_games["win"] = (all_games["wl"] == "W").astype(int)
    all_games["stl_blk"] = all_games["stl"] + all_games["blk"]
    
    # Compute expanding averages (excludes current game)
    groupby_cols = ["season_id", "team_id"]
    
    # Shift by 1 to avoid data leakage (we want stats BEFORE this game)
    all_games["win_pct"] = all_games.groupby(groupby_cols)["win"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["ppg"] = all_games.groupby(groupby_cols)["pts"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["opp_ppg"] = all_games.groupby(groupby_cols)["opp_pts"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["fg_pct_avg"] = all_games.groupby(groupby_cols)["fg_pct"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["fg3_pct_avg"] = all_games.groupby(groupby_cols)["fg3_pct"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["ft_pct_avg"] = all_games.groupby(groupby_cols)["ft_pct"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["rpg"] = all_games.groupby(groupby_cols)["reb"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["apg"] = all_games.groupby(groupby_cols)["ast"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["topg"] = all_games.groupby(groupby_cols)["tov"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_games["stl_blk_pg"] = all_games.groupby(groupby_cols)["stl_blk"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Merge back to original dataframe
    home_rolling = all_games[all_games["is_home"] == 1][[
        "game_id", "win_pct", "ppg", "opp_ppg", "fg_pct_avg", "fg3_pct_avg",
        "ft_pct_avg", "rpg", "apg", "topg", "stl_blk_pg"
    ]].copy()
    home_rolling.columns = [
        "game_id", "home_win_pct", "home_ppg", "home_opp_ppg", "home_fg_pct",
        "home_fg3_pct", "home_ft_pct", "home_rpg", "home_apg", "home_topg", "home_stl_blk_pg"
    ]
    
    away_rolling = all_games[all_games["is_home"] == 0][[
        "game_id", "win_pct", "ppg", "opp_ppg", "fg_pct_avg", "fg3_pct_avg",
        "ft_pct_avg", "rpg", "apg", "topg", "stl_blk_pg"
    ]].copy()
    away_rolling.columns = [
        "game_id", "away_win_pct", "away_ppg", "away_opp_ppg", "away_fg_pct",
        "away_fg3_pct", "away_ft_pct", "away_rpg", "away_apg", "away_topg", "away_stl_blk_pg"
    ]
    
    df = df.merge(home_rolling, on="game_id", how="left")
    df = df.merge(away_rolling, on="game_id", how="left")
    
    return df


# =============================================================================
# SECTION 3: RECENT FORM FEATURES (Last 5 and 10 games)
# =============================================================================

def compute_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute recent form metrics based on last N games.
    
    Features:
    - Win percentage in last 5 and 10 games
    - Scoring trend (PPG change in last 5 vs previous 5)
    """
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    
    # Create long format again
    home_games = df[[
        "game_id", "game_date", "season_id", "team_id_home", "wl_home", "pts_home"
    ]].copy()
    home_games.columns = ["game_id", "game_date", "season_id", "team_id", "wl", "pts"]
    home_games["is_home"] = 1
    
    away_games = df[[
        "game_id", "game_date", "season_id", "team_id_away", "wl_away", "pts_away"
    ]].copy()
    away_games.columns = ["game_id", "game_date", "season_id", "team_id", "wl", "pts"]
    away_games["is_home"] = 0
    
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(["season_id", "team_id", "game_date"]).reset_index(drop=True)
    
    all_games["win"] = (all_games["wl"] == "W").astype(int)
    
    # Compute rolling averages for last 5 and 10 games (excluding current game)
    groupby_cols = ["season_id", "team_id"]
    
    all_games["last5_win_pct"] = all_games.groupby(groupby_cols)["win"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    all_games["last10_win_pct"] = all_games.groupby(groupby_cols)["win"].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    
    # Scoring trend: compare last 5 games to previous 5 games
    all_games["ppg_last5"] = all_games.groupby(groupby_cols)["pts"].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    all_games["ppg_prev5"] = all_games.groupby(groupby_cols)["pts"].transform(
        lambda x: x.shift(6).rolling(window=5, min_periods=1).mean()
    )
    all_games["scoring_trend"] = all_games["ppg_last5"] - all_games["ppg_prev5"]
    
    # Merge back
    home_form = all_games[all_games["is_home"] == 1][[
        "game_id", "last5_win_pct", "last10_win_pct", "scoring_trend"
    ]].copy()
    home_form.columns = ["game_id", "home_last5_win_pct", "home_last10_win_pct", "home_scoring_trend"]
    
    away_form = all_games[all_games["is_home"] == 0][[
        "game_id", "last5_win_pct", "last10_win_pct", "scoring_trend"
    ]].copy()
    away_form.columns = ["game_id", "away_last5_win_pct", "away_last10_win_pct", "away_scoring_trend"]
    
    df = df.merge(home_form, on="game_id", how="left")
    df = df.merge(away_form, on="game_id", how="left")
    
    return df


# =============================================================================
# SECTION 4: REST & SCHEDULE CONTEXT
# =============================================================================

def compute_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest-related features:
    - Days since last game for each team
    - Back-to-back indicators
    - Categorized rest days
    """
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    
    # Create long format
    home_games = df[["game_id", "game_date", "season_id", "team_id_home"]].copy()
    home_games.columns = ["game_id", "game_date", "season_id", "team_id"]
    home_games["is_home"] = 1
    
    away_games = df[["game_id", "game_date", "season_id", "team_id_away"]].copy()
    away_games.columns = ["game_id", "game_date", "season_id", "team_id"]
    away_games["is_home"] = 0
    
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(["season_id", "team_id", "game_date"]).reset_index(drop=True)
    
    # Calculate days since last game
    all_games["prev_game_date"] = all_games.groupby(["season_id", "team_id"])["game_date"].shift(1)
    all_games["days_rest"] = (all_games["game_date"] - all_games["prev_game_date"]).dt.days
    all_games["days_rest"] = all_games["days_rest"].fillna(7)  # First game of season = well rested
    
    # Back-to-back indicator
    all_games["back_to_back"] = (all_games["days_rest"] == 1).astype(int)
    
    # Categorize rest days
    all_games["days_rest_cat"] = pd.cut(
        all_games["days_rest"],
        bins=[-1, 1, 2, 3, 100],
        labels=["B2B", "1_day", "2_days", "3+_days"]
    )
    
    # Merge back
    home_rest = all_games[all_games["is_home"] == 1][[
        "game_id", "days_rest", "back_to_back", "days_rest_cat"
    ]].copy()
    home_rest.columns = ["game_id", "days_rest_home", "back_to_back_home", "days_rest_home_cat"]
    
    away_rest = all_games[all_games["is_home"] == 0][[
        "game_id", "days_rest", "back_to_back", "days_rest_cat"
    ]].copy()
    away_rest.columns = ["game_id", "days_rest_away", "back_to_back_away", "days_rest_away_cat"]
    
    df = df.merge(home_rest, on="game_id", how="left")
    df = df.merge(away_rest, on="game_id", how="left")
    
    return df


# =============================================================================
# SECTION 5: ROSTER HEALTH
# =============================================================================

def add_roster_health(df: pd.DataFrame, inactive_players: pd.DataFrame) -> pd.DataFrame:
    """
    Add roster health features based on number of inactive players.
    
    Features:
    - Number of inactive players per team
    - Categorized roster health (Full vs Depleted)
    """
    # Count inactive players per game per team
    inactive_counts = inactive_players.groupby(["game_id", "team_id"]).size().reset_index(name="num_inactive")
    
    # Merge for home team
    home_inactive = inactive_counts.copy()
    home_inactive.columns = ["game_id", "team_id_home", "num_inactive_home"]
    df = df.merge(home_inactive, on=["game_id", "team_id_home"], how="left")
    
    # Merge for away team
    away_inactive = inactive_counts.copy()
    away_inactive.columns = ["game_id", "team_id_away", "num_inactive_away"]
    df = df.merge(away_inactive, on=["game_id", "team_id_away"], how="left")
    
    # Fill NaN with 0 (no inactive players)
    df["num_inactive_home"] = df["num_inactive_home"].fillna(0).astype(int)
    df["num_inactive_away"] = df["num_inactive_away"].fillna(0).astype(int)
    
    # Categorize roster health
    df["roster_health_home"] = pd.cut(
        df["num_inactive_home"],
        bins=[-1, 1, 100],
        labels=["Full", "Depleted"]
    )
    df["roster_health_away"] = pd.cut(
        df["num_inactive_away"],
        bins=[-1, 1, 100],
        labels=["Full", "Depleted"]
    )
    
    return df


# =============================================================================
# SECTION 6: ADVANCED STATS FROM OTHER_STATS
# =============================================================================

def add_advanced_stats(df: pd.DataFrame, other_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced statistics from other_stats.csv:
    - Points in paint per game
    - Fast break points per game
    
    These are computed as rolling averages before each game.
    """
    # Merge other_stats to get current game advanced stats
    df = df.merge(
        other_stats[["game_id", "pts_paint_home", "pts_fb_home", "pts_paint_away", "pts_fb_away"]],
        on="game_id",
        how="left"
    )
    
    # Create long format to compute rolling averages
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    
    home_advanced = df[[
        "game_id", "game_date", "season_id", "team_id_home", "pts_paint_home", "pts_fb_home"
    ]].copy()
    home_advanced.columns = ["game_id", "game_date", "season_id", "team_id", "pts_paint", "pts_fb"]
    home_advanced["is_home"] = 1
    
    away_advanced = df[[
        "game_id", "game_date", "season_id", "team_id_away", "pts_paint_away", "pts_fb_away"
    ]].copy()
    away_advanced.columns = ["game_id", "game_date", "season_id", "team_id", "pts_paint", "pts_fb"]
    away_advanced["is_home"] = 0
    
    all_advanced = pd.concat([home_advanced, away_advanced], ignore_index=True)
    all_advanced = all_advanced.sort_values(["season_id", "team_id", "game_date"]).reset_index(drop=True)
    
    # Compute expanding averages (excluding current game)
    groupby_cols = ["season_id", "team_id"]
    all_advanced["paint_pts_pg"] = all_advanced.groupby(groupby_cols)["pts_paint"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    all_advanced["fastbreak_pts_pg"] = all_advanced.groupby(groupby_cols)["pts_fb"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Merge back
    home_adv = all_advanced[all_advanced["is_home"] == 1][[
        "game_id", "paint_pts_pg", "fastbreak_pts_pg"
    ]].copy()
    home_adv.columns = ["game_id", "home_paint_pts_pg", "home_fastbreak_pts_pg"]
    
    away_adv = all_advanced[all_advanced["is_home"] == 0][[
        "game_id", "paint_pts_pg", "fastbreak_pts_pg"
    ]].copy()
    away_adv.columns = ["game_id", "away_paint_pts_pg", "away_fastbreak_pts_pg"]
    
    df = df.merge(home_adv, on="game_id", how="left")
    df = df.merge(away_adv, on="game_id", how="left")
    
    # Drop the current game advanced stats (we only want historical averages)
    df = df.drop(columns=["pts_paint_home", "pts_fb_home", "pts_paint_away", "pts_fb_away"], errors="ignore")
    
    return df


# =============================================================================
# SECTION 7: GAME CONTEXT FEATURES
# =============================================================================

def add_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game context features:
    - Home court advantage (always 1 for regular games)
    - Season stage (Early/Mid/Late)
    - Weekend indicator
    """
    df = df.copy()
    
    # Home court advantage (always 1 for regular season games)
    df["home_court_advantage"] = 1
    
    # Calculate game number in season for each team
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    df["home_game_num"] = df.groupby(["season_id", "team_id_home"]).cumcount() + 1
    
    # Season stage based on game number
    df["season_stage"] = pd.cut(
        df["home_game_num"],
        bins=[0, 30, 60, 100],
        labels=["Early", "Mid", "Late"]
    )
    
    # Weekend indicator (Saturday = 5, Sunday = 6)
    df["is_weekend"] = df["game_date"].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Drop temporary column
    df = df.drop(columns=["home_game_num"], errors="ignore")
    
    return df


# =============================================================================
# SECTION 8: DERIVED/MATCHUP FEATURES
# =============================================================================

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that capture matchup dynamics:
    - Strength differential
    - Offensive advantage
    - Defensive advantage
    - Rest advantage
    - Form difference
    - Overall matchup advantage category
    """
    df = df.copy()
    
    # Strength differential
    df["strength_differential"] = df["home_win_pct"] - df["away_win_pct"]
    
    # Offensive advantage (how well home offense matches up with away defense)
    df["offensive_advantage"] = df["home_ppg"] - df["away_opp_ppg"]
    
    # Defensive advantage (how well home defense matches up with away offense)
    df["defensive_advantage"] = df["away_opp_ppg"] - df["home_opp_ppg"]
    
    # Rest advantage
    df["rest_advantage"] = df["days_rest_home"] - df["days_rest_away"]
    
    # Form difference
    df["form_difference"] = df["home_last5_win_pct"] - df["away_last5_win_pct"]
    
    # Overall matchup advantage (discretized)
    # Based on combination of strength differential and form
    df["combined_advantage"] = df["strength_differential"] + 0.5 * df["form_difference"]
    df["matchup_advantage"] = pd.cut(
        df["combined_advantage"],
        bins=[-np.inf, -0.1, 0.1, np.inf],
        labels=["Away_Favored", "Even", "Home_Favored"]
    )
    
    # Drop temporary column
    df = df.drop(columns=["combined_advantage"], errors="ignore")
    
    return df


# =============================================================================
# SECTION 9: DISCRETIZATION FOR BAYESIAN NETWORKS
# =============================================================================

def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create discretized versions of continuous features for discrete Bayesian Networks.
    
    Discretized features:
    - Team strength (based on win%)
    - Offensive strength (based on PPG)
    - Defensive strength (based on opponent PPG)
    - Recent form (based on last 5 games)
    """
    df = df.copy()
    
    # Team strength based on win percentage
    for side in ["home", "away"]:
        col = f"{side}_win_pct"
        df[f"{side}_team_strength"] = pd.cut(
            df[col],
            bins=[-np.inf, 0.4, 0.5, 0.6, np.inf],
            labels=["Weak", "Below_Avg", "Above_Avg", "Strong"]
        )
    
    # Offensive strength based on PPG (using percentiles)
    for side in ["home", "away"]:
        col = f"{side}_ppg"
        df[f"{side}_offensive_strength"] = pd.qcut(
            df[col],
            q=4,
            labels=["Poor", "Average", "Good", "Elite"],
            duplicates="drop"
        )
    
    # Defensive strength based on opponent PPG (lower is better)
    for side in ["home", "away"]:
        col = f"{side}_opp_ppg"
        df[f"{side}_defensive_strength"] = pd.qcut(
            df[col],
            q=4,
            labels=["Elite", "Good", "Average", "Poor"],  # Reversed because lower is better
            duplicates="drop"
        )
    
    # Recent form based on last 5 games
    for side in ["home", "away"]:
        col = f"{side}_last5_win_pct"
        df[f"{side}_recent_form"] = pd.cut(
            df[col],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=["Poor", "Average", "Good"]
        )
    
    return df


# =============================================================================
# SECTION 10: DROP EARLY GAMES (AFTER COMPUTING ALL FEATURES)
# =============================================================================

def drop_early_games(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Drop first N games of each team's season.
    This is done AFTER computing rolling statistics to ensure we have enough data.
    """
    df = df.sort_values(["season_id", "team_id_home", "game_date"]).reset_index(drop=True)
    
    df["home_game_number"] = df.groupby(["season_id", "team_id_home"]).cumcount() + 1
    df = df.sort_values(["season_id", "team_id_away", "game_date"]).reset_index(drop=True)
    df["away_game_number"] = df.groupby(["season_id", "team_id_away"]).cumcount() + 1
    
    # Keep only games where BOTH teams have played more than N games
    keep = (df["home_game_number"] > n) & (df["away_game_number"] > n)
    df = df.loc[keep].drop(columns=["home_game_number", "away_game_number"])
    
    return df.reset_index(drop=True)


# =============================================================================
# SECTION 11: MAIN PROCESSING PIPELINE
# =============================================================================

def make_processed_games_full(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Main processing pipeline that applies all transformations.
    
    Args:
        raw: Dictionary containing raw dataframes:
            - 'games': game.csv
            - 'other_stats': other_stats.csv
            - 'game_info': game_info.csv
            - 'inactive_players': inactive_players.csv
    
    Returns:
        Fully processed dataframe with all features
    """
    print("Starting comprehensive preprocessing pipeline...")
    
    games = raw["games"]
    other_stats = raw["other_stats"]
    game_info = raw["game_info"]
    inactive_players = raw["inactive_players"]
    
    # Step 1: Basic filtering
    print("Step 1: Filtering to modern era and regular season...")
    df = filter_modern_regular_season(games)
    print(f"  → {len(df):,} games after filtering")
    
    # Step 2: Add target variable
    print("Step 2: Adding target variable...")
    df = add_target_variable(df)
    
    # Step 3: Merge game_info for additional context
    print("Step 3: Merging game info...")
    if "game_date" not in game_info.columns or game_info["game_date"].dtype == "object":
        game_info["game_date"] = pd.to_datetime(game_info["game_date"])
    df = df.merge(
        game_info[["game_id", "attendance", "game_time"]],
        on="game_id",
        how="left"
    )
    
    # Step 4: Compute rolling statistics
    print("Step 4: Computing rolling statistics...")
    df = compute_rolling_stats(df)
    
    # Step 5: Compute recent form
    print("Step 5: Computing recent form features...")
    df = compute_recent_form(df)
    
    # Step 6: Compute rest features
    print("Step 6: Computing rest and schedule features...")
    df = compute_rest_features(df)
    
    # Step 7: Add roster health
    print("Step 7: Adding roster health features...")
    df = add_roster_health(df, inactive_players)
    
    # Step 8: Add advanced stats
    print("Step 8: Adding advanced statistics...")
    df = add_advanced_stats(df, other_stats)
    
    # Step 9: Add game context
    print("Step 9: Adding game context features...")
    df = add_game_context(df)
    
    # Step 10: Add derived features
    print("Step 10: Computing derived and matchup features...")
    df = add_derived_features(df)
    
    # Step 11: Discretize features for Bayesian Networks
    print("Step 11: Discretizing features for Bayesian Networks...")
    df = discretize_features(df)
    
    # Step 12: Drop early season games (AFTER computing all features)
    print(f"Step 12: Dropping first {N_EARLY_GAMES_DROP} games per team...")
    df = drop_early_games(df, N_EARLY_GAMES_DROP)
    print(f"  → {len(df):,} games after dropping early games")
    
    # Final cleanup
    print("Step 13: Final cleanup...")
    df = df.sort_values("game_date").reset_index(drop=True)
    
    print(f"\nPreprocessing complete!")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    
    return df


def save_processed_games(df: pd.DataFrame, filename: str = "games_processed_full.parquet") -> None:
    """Save the fully processed dataframe to parquet format."""
    out_dir = Path(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    df.to_parquet(output_path, index=False)
    print(f"Saved processed data to: {output_path}")


def get_feature_summary(df: pd.DataFrame) -> None:
    """Print a summary of all features in the processed dataframe."""
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    
    # Categorize columns
    id_cols = [c for c in df.columns if "id" in c.lower() or c == "game_date"]
    target_col = ["game_outcome"]
    continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    continuous_cols = [c for c in continuous_cols if c not in id_cols + target_col]
    categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
    bool_cols = df.select_dtypes(include=[bool]).columns.tolist()
    
    print(f"\nIdentifiers ({len(id_cols)}): {', '.join(id_cols)}")
    print(f"\nTarget variable (1): {target_col[0]}")
    print(f"\nContinuous features ({len(continuous_cols)}):")
    for i, col in enumerate(continuous_cols, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nCategorical features ({len(categorical_cols)}):")
    for i, col in enumerate(categorical_cols, 1):
        print(f"  {i:2d}. {col}")
    print(f"\nBoolean features ({len(bool_cols)}):")
    for i, col in enumerate(bool_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n{'='*80}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Total rows: {df.shape[0]:,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("="*80 + "\n")


# =============================================================================
# SECTION 12: TRAIN/VAL/TEST SPLIT
# =============================================================================

def split_train_val_test(
    df: pd.DataFrame,
    train_end_season: int = 2018,
    val_end_season: int = 2021
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets based on season.
    
    Args:
        df: Processed dataframe
        train_end_season: Last season for training (inclusive)
        val_end_season: Last season for validation (inclusive)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train = df[df["season_id"] <= train_end_season].copy()
    val = df[(df["season_id"] > train_end_season) & (df["season_id"] <= val_end_season)].copy()
    test = df[df["season_id"] > val_end_season].copy()
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train):,} games (seasons ≤ {train_end_season})")
    print(f"  Validation: {len(val):,} games (seasons {train_end_season+1}-{val_end_season})")
    print(f"  Test:       {len(test):,} games (seasons > {val_end_season})")
    
    return train, val, test

