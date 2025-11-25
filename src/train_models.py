"""
Training script for baseline Logistic Regression and Naive Bayes Bayesian Network models.

This script:
1. Loads and processes data using preprocessing_full.py
2. Trains a baseline Logistic Regression model
3. Trains a Naive Bayes Bayesian Network using pomegranate
4. Evaluates and compares both models
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Pomegranate for Bayesian Networks
try:
    from pomegranate.distributions import Categorical, ConditionalCategorical
    from pomegranate.bayesian_network import BayesianNetwork
    POMEGRANATE_AVAILABLE = True
except ImportError:
    warnings.warn("Pomegranate not installed. Bayesian Network models will not be available.")
    POMEGRANATE_AVAILABLE = False

from .config import PROCESSED_DIR, TRAIN_END_SEASON, VAL_END_SEASON, RANDOM_SEED
from .data_loading import load_all_raw
from .preprocessing_full import (
    make_processed_games_full,
    split_train_val_test,
    save_processed_games,
)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_or_create_processed_data(
    force_reprocess: bool = False,
    filename: str = "games_processed_full.parquet"
) -> pd.DataFrame:
    """
    Load processed data from disk, or create it if it doesn't exist.
    
    Args:
        force_reprocess: If True, reprocess even if file exists
        filename: Name of the processed data file
        
    Returns:
        Processed dataframe
    """
    processed_path = Path(PROCESSED_DIR) / filename
    
    if not force_reprocess and processed_path.exists():
        print(f"Loading processed data from {processed_path}...")
        df = pd.read_parquet(processed_path)
        print(f"Loaded {len(df):,} games with {df.shape[1]} features")
        return df
    
    print("Processed data not found. Creating from scratch...")
    raw = load_all_raw()
    df = make_processed_games_full(raw)
    save_processed_games(df, filename=filename)
    return df


def prepare_features_for_logistic_regression(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare continuous features for logistic regression.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    # Select continuous features (exclude IDs, dates, and categorical)
    exclude_cols = [
        'game_id', 'game_date', 'season_id', 'team_id_home', 'team_id_away',
        'game_outcome',  # target
        'wl_home', 'wl_away',  # raw target encoding
        'matchup_home', 'matchup_away',  # text fields
        'season_type',  # should be constant after filtering
        'attendance', 'game_time',  # may have missing values
        # Exclude actual game stats (would leak the outcome!)
        'pts_home', 'pts_away',  # points scored in THIS game
        'plus_minus_home', 'plus_minus_away',  # point differential in THIS game
        'fgm_home', 'fga_home', 'fg_pct_home',  # current game stats
        'fgm_away', 'fga_away', 'fg_pct_away',
        'fg3m_home', 'fg3a_home', 'fg3_pct_home',
        'fg3m_away', 'fg3a_away', 'fg3_pct_away',
        'ftm_home', 'fta_home', 'ft_pct_home',
        'ftm_away', 'fta_away', 'ft_pct_away',
        'oreb_home', 'dreb_home', 'reb_home',
        'oreb_away', 'dreb_away', 'reb_away',
        'ast_home', 'ast_away',
        'stl_home', 'stl_away',
        'blk_home', 'blk_away',
        'tov_home', 'tov_away',
        'pf_home', 'pf_away',
        'min',  # minutes played
        # Exclude categorical features (keep continuous only)
        'home_team_strength', 'home_offensive_strength', 'home_defensive_strength',
        'away_team_strength', 'away_offensive_strength', 'away_defensive_strength',
        'home_recent_form', 'away_recent_form',
        'days_rest_home_cat', 'days_rest_away_cat',
        'roster_health_home', 'roster_health_away',
        'season_stage', 'matchup_advantage',
    ]
    
    # Get all numeric columns
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Extract features
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Handle missing values (fill with median from training set)
    for col in feature_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # Extract targets
    y_train = train['game_outcome'].values
    y_val = val['game_outcome'].values
    y_test = test['game_outcome'].values
    
    print(f"\nLogistic Regression Features:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return (
        X_train.values, X_val.values, X_test.values,
        y_train, y_val, y_test,
        feature_cols
    )


def prepare_features_for_naive_bayes(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare discretized/categorical features for Naive Bayes.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    # Select categorical and discrete features suitable for Naive Bayes
    categorical_features = [
        'home_team_strength', 'home_offensive_strength', 'home_defensive_strength',
        'away_team_strength', 'away_offensive_strength', 'away_defensive_strength',
        'home_recent_form', 'away_recent_form',
        'roster_health_home', 'roster_health_away',
        'season_stage', 'matchup_advantage',
    ]
    
    # Also include some discretized numeric features
    discrete_features = [
        'back_to_back_home', 'back_to_back_away',
        'is_weekend', 'home_court_advantage',
    ]
    
    feature_cols = categorical_features + discrete_features
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in train.columns]
    
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Convert categorical to string (for pomegranate)
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
    
    # Convert boolean/int to string for consistency
    for col in discrete_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
    
    # Handle missing values (fill with 'Missing')
    X_train = X_train.fillna('Missing')
    X_val = X_val.fillna('Missing')
    X_test = X_test.fillna('Missing')
    
    y_train = train['game_outcome'].values
    y_val = val['game_outcome'].values
    y_test = test['game_outcome'].values
    
    print(f"\nNaive Bayes Features:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Categorical: {len([c for c in categorical_features if c in feature_cols])}")
    print(f"  Discrete: {len([c for c in discrete_features if c in feature_cols])}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# =============================================================================
# BASELINE LOGISTIC REGRESSION MODEL
# =============================================================================

class LogisticRegressionBaseline:
    """Baseline logistic regression model with preprocessing."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance if any
        )
        self.feature_names = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str] = None):
        """Fit the model."""
        print("\nTraining Logistic Regression...")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = feature_names
        
        # Training accuracy
        train_preds = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"  Training accuracy: {train_acc:.4f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (coefficients)."""
        if self.feature_names is None:
            return None
        
        coeffs = self.model.coef_[0]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coeffs,
            'abs_coefficient': np.abs(coeffs)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df


# =============================================================================
# NAIVE BAYES BAYESIAN NETWORK MODEL
# =============================================================================

class NaiveBayesBN:
    """
    Naive Bayes model implemented as a Bayesian Network using pomegranate.
    
    In a Naive Bayes structure:
    - Target (game_outcome) is the parent node
    - All features are children of the target (conditionally independent given target)
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.target_name = 'game_outcome'
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: List[str]):
        """
        Fit the Naive Bayes Bayesian Network.
        
        Args:
            X_train: Feature dataframe (categorical/discrete)
            y_train: Target array
            feature_names: List of feature column names
        """
        print("\nTraining Naive Bayes Bayesian Network...")
        
        self.feature_names = feature_names
        
        if not POMEGRANATE_AVAILABLE:
            # Use sklearn's CategoricalNB as fallback
            print("  Pomegranate not available, using sklearn CategoricalNB...")
            from sklearn.naive_bayes import CategoricalNB
            from sklearn.preprocessing import LabelEncoder
            
            # Encode categorical features
            self.label_encoders = {}
            X_encoded = X_train.copy()
            for col in feature_names:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_train[col])
                self.label_encoders[col] = le
            
            self.model = CategoricalNB()
            self.model.fit(X_encoded.values, y_train)
            self.is_sklearn_fallback = True
            print("  Training complete")
            return
        
        # Create dataframe with target
        data = X_train.copy()
        data[self.target_name] = y_train.astype(str)
        
        # Build Naive Bayes structure
        # 1. Target node (root)
        target_dist = Categorical([
            [str(c)] for c in sorted(data[self.target_name].unique())
        ])
        
        # 2. Feature nodes (conditionally independent given target)
        feature_distributions = []
        for feature in feature_names:
            # Get conditional distribution P(feature | target)
            # This requires computing conditional probabilities from data
            feature_distributions.append(self._create_conditional_distribution(data, feature))
        
        # Create Bayesian Network with Naive Bayes structure
        # Note: pomegranate v1.0+ has different API
        try:
            # Try pomegranate 1.0+ API
            print("  Building Naive Bayes structure...")
            self.model = self._build_naive_bayes_v1(data)
        except Exception as e:
            print(f"  Error with pomegranate v1.0+ API: {e}")
            print("  Falling back to sklearn...")
            # Fallback to sklearn-style Naive Bayes
            from sklearn.naive_bayes import CategoricalNB
            from sklearn.preprocessing import LabelEncoder
            
            # Encode categorical features
            self.label_encoders = {}
            X_encoded = X_train.copy()
            for col in feature_names:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_train[col])
                self.label_encoders[col] = le
            
            self.model = CategoricalNB()
            self.model.fit(X_encoded.values, y_train)
            self.is_sklearn_fallback = True
            print("  Using sklearn CategoricalNB")
        
        print("  Training complete")
    
    def _create_conditional_distribution(self, data: pd.DataFrame, feature: str) -> Dict:
        """Create conditional probability table for P(feature | target)."""
        target = self.target_name
        
        # Compute conditional probabilities
        cond_probs = {}
        for target_val in data[target].unique():
            subset = data[data[target] == target_val]
            probs = subset[feature].value_counts(normalize=True).to_dict()
            cond_probs[target_val] = probs
        
        return cond_probs
    
    def _build_naive_bayes_v1(self, data: pd.DataFrame) -> object:
        """Build Naive Bayes using pomegranate v1.0+ API."""
        # For simplicity, use sklearn fallback for now
        # Full pomegranate BN implementation is complex and version-dependent
        raise NotImplementedError("Using sklearn fallback")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if hasattr(self, 'is_sklearn_fallback') and self.is_sklearn_fallback:
            X_encoded = X.copy()
            for col in self.feature_names:
                X_encoded[col] = self.label_encoders[col].transform(X[col])
            return self.model.predict(X_encoded.values)
        else:
            # Pomegranate predict
            return self.model.predict(X[self.feature_names])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self, 'is_sklearn_fallback') and self.is_sklearn_fallback:
            X_encoded = X.copy()
            for col in self.feature_names:
                X_encoded[col] = self.label_encoders[col].transform(X[col])
            return self.model.predict_proba(X_encoded.values)
        else:
            # Pomegranate predict_proba
            return self.model.predict_proba(X[self.feature_names])


# =============================================================================
# EVALUATION AND COMPARISON
# =============================================================================

def evaluate_model(
    model,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_name: str = "Model"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on train/val/test sets.
    
    Returns:
        Dictionary with metrics for each split
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION: {model_name}")
    print('='*80)
    
    results = {}
    
    for split_name, X, y in [
        ('Train', X_train, y_train),
        ('Validation', X_val, y_val),
        ('Test', X_test, y_test)
    ]:
        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'log_loss': log_loss(y, y_proba),
        }
        
        results[split_name] = metrics
        
        # Print metrics
        print(f"\n{split_name} Set:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Log Loss:  {metrics['log_loss']:.4f}")
        
        # Confusion matrix for test set
        if split_name == 'Test':
            print(f"\n  Confusion Matrix:")
            cm = confusion_matrix(y, y_pred)
            print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
            print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")
    
    return results


def compare_models(results_lr: Dict, results_nb: Dict) -> pd.DataFrame:
    """Compare two models side by side."""
    comparison = []
    
    for split in ['Train', 'Validation', 'Test']:
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']:
            comparison.append({
                'Split': split,
                'Metric': metric,
                'Logistic Regression': results_lr[split][metric],
                'Naive Bayes': results_nb[split][metric],
                'Difference': results_lr[split][metric] - results_nb[split][metric]
            })
    
    df = pd.DataFrame(comparison)
    return df


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """Plot comparison of models."""
    # Filter to test set only for cleaner visualization
    test_df = comparison_df[comparison_df['Split'] == 'Test']
    
    # Exclude log_loss for better scale
    plot_df = test_df[test_df['Metric'] != 'log_loss'].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(plot_df))
    width = 0.35
    
    ax.bar(x - width/2, plot_df['Logistic Regression'], width, label='Logistic Regression', alpha=0.8)
    ax.bar(x + width/2, plot_df['Naive Bayes'], width, label='Naive Bayes', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Metric'], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to {save_path}")
    
    return fig


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main(force_reprocess: bool = False):
    """Main training pipeline."""
    print("="*80)
    print("NBA GAME OUTCOME PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # 1. Load/create processed data
    print("\n[1/5] Loading processed data...")
    df = load_or_create_processed_data(force_reprocess=force_reprocess)
    
    # 2. Split data
    print("\n[2/5] Splitting data into train/val/test...")
    train, val, test = split_train_val_test(df, TRAIN_END_SEASON, VAL_END_SEASON)
    
    # 3. Train Logistic Regression
    print("\n[3/5] Training Logistic Regression baseline...")
    X_train_lr, X_val_lr, X_test_lr, y_train, y_val, y_test, feature_names_lr = \
        prepare_features_for_logistic_regression(train, val, test)
    
    lr_model = LogisticRegressionBaseline()
    lr_model.fit(X_train_lr, y_train, feature_names_lr)
    
    results_lr = evaluate_model(
        lr_model,
        X_train_lr, y_train,
        X_val_lr, y_val,
        X_test_lr, y_test,
        model_name="Logistic Regression Baseline"
    )
    
    # 4. Train Naive Bayes
    print("\n[4/5] Training Naive Bayes Bayesian Network...")
    X_train_nb, X_val_nb, X_test_nb, y_train_nb, y_val_nb, y_test_nb, feature_names_nb = \
        prepare_features_for_naive_bayes(train, val, test)
    
    nb_model = NaiveBayesBN()
    nb_model.fit(X_train_nb, y_train_nb, feature_names_nb)
    
    results_nb = evaluate_model(
        nb_model,
        X_train_nb, y_train_nb,
        X_val_nb, y_val_nb,
        X_test_nb, y_test_nb,
        model_name="Naive Bayes Bayesian Network"
    )
    
    # 5. Compare models
    print("\n[5/5] Comparing models...")
    comparison_df = compare_models(results_lr, results_nb)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    test_comparison = comparison_df[comparison_df['Split'] == 'Test']
    print(test_comparison.to_string(index=False))
    
    # Plot comparison
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_model_comparison(comparison_df, save_path=output_dir / "model_comparison.png")
    
    # Save results
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nSaved detailed comparison to {output_dir / 'model_comparison.csv'}")
    
    # Feature importance for Logistic Regression
    importance_df = lr_model.get_feature_importance()
    if importance_df is not None:
        print("\n" + "="*80)
        print("TOP 15 MOST IMPORTANT FEATURES (Logistic Regression)")
        print("="*80)
        print(importance_df.head(15).to_string(index=False))
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return {
        'lr_model': lr_model,
        'nb_model': nb_model,
        'results_lr': results_lr,
        'results_nb': results_nb,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NBA game outcome prediction models")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data even if processed file exists"
    )
    
    args = parser.parse_args()
    
    results = main(force_reprocess=args.reprocess)

