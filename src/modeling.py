import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from pomegranate import NaiveBayes, DiscreteDistribution

from .config import (
    TRAIN_END_SEASON,
    VAL_END_SEASON,
    TEST_END_SEASON,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


CANDIDATE_FEATURES = [
    "fg_pct_home", "fg_pct_away",
    "fg3_pct_home", "fg3_pct_away",
    "ft_pct_home", "ft_pct_away",
    "reb_home", "reb_away",
    "ast_home", "ast_away",
    "stl_home", "stl_away",
    "blk_home", "blk_away",
    "tov_home", "tov_away",
    "pts_home", "pts_away",
    "pts_paint_home", "pts_paint_away",
]

TARGET_COL = "home_win"


def build_feature_table(games_df: pd.DataFrame):
    df = games_df.copy()

    if TARGET_COL not in df.columns:
        df[TARGET_COL] = (df["wl_home"] == "W").astype(int)

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]

    keep_cols = [c for c in ["season_id", "game_id", "game_date"] if c in df.columns]
    cols = keep_cols + feature_cols + [TARGET_COL]

    df = df[cols].dropna().reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df, feature_cols


def split_by_season(table, feature_cols):
    years = pd.to_datetime(table["game_date"]).dt.year

    train_mask = years <= TRAIN_END_SEASON
    val_mask = (years > TRAIN_END_SEASON) & (years <= VAL_END_SEASON)
    test_mask = (years > VAL_END_SEASON) & (years <= TEST_END_SEASON)

    def xy(mask):
        df = table.loc[mask].reset_index(drop=True)
        X = df[feature_cols].to_numpy()
        y = df[TARGET_COL].to_numpy()
        return X, y, df

    return xy(train_mask), xy(val_mask), xy(test_mask)


def train_logistic_baseline(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def predict_logistic_proba(model, X):
    return model.predict_proba(X)[:, 1]


# -------------------- FIXED NAIVE BAYES BN (compatible with your version) -------------------- #

def train_bn_model(train_df, feature_cols):
    X_num = train_df[feature_cols].to_numpy()
    y = train_df[TARGET_COL].astype(int).to_numpy()

    disc = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    X_disc = disc.fit_transform(X_num).astype(int)

    # >>> FIX: old API uses positional y, not labels= <<< #
    nb = NaiveBayes.from_samples(
        DiscreteDistribution,
        X_disc,
        y
    )

    return nb, disc, feature_cols


def predict_bn_proba(nb, df, feature_cols, disc):
    X_num = df[feature_cols].to_numpy()
    X_disc = disc.transform(X_num).astype(int)

    proba_list = nb.predict_proba(X_disc)

    result = []
    for p in proba_list:
        # pomegranate old API returns dict-like distributions
        if hasattr(p, "parameters"):
            params = p.parameters[0]
            result.append(params.get(1, 0.5))
        else:
            result.append(p[1])  # fallback for array-like

    return np.array(result)


# -------------------- Evaluation -------------------- #

def evaluate_model(name, y_true, proba):
    y_pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)

    proba_two = np.vstack([1 - proba, proba]).T

    try:
        ll = log_loss(y_true, proba_two)
    except ValueError:
        ll = np.nan

    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan

    print(f"{name}: acc={acc:.3f}, logloss={ll:.3f}, auc={auc:.3f}")

    return {"accuracy": acc, "log_loss": ll, "roc_auc": auc}
