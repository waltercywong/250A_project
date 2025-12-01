"""
Refined Hierarchical Bayesian Network with improvements:

1. Better discretization (5-6 bins, percentile-based)
2. Simplified structure (3 layers vs 5, direct connections)
3. Key features (strength_differential, interaction features)
4. Feature selection (top predictive features only)

Structure:
    Layer 1 (Root nodes):
        home_strength, away_strength, days_rest_home, days_rest_away
    
    Layer 2 (Derived features):
        strength_differential, offensive_matchup, home_form, away_form
    
    Layer 3 (Outcome):
        game_outcome (with direct connections from key predictors)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
)

# pgmpy for Bayesian Networks
try:
    from pgmpy.models import BayesianNetwork as BayesianNetworkOld
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        BayesianNetwork = BayesianNetworkOld
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not installed. Install with: pip install pgmpy")

from .config import PROCESSED_DIR, TRAIN_END_SEASON, VAL_END_SEASON, RANDOM_SEED
from .data_loading import load_all_raw
from .preprocessing_full import (
    make_processed_games_full,
    split_train_val_test,
    save_processed_games,
)
from .train_models import (
    load_or_create_processed_data,
    LogisticRegressionBaseline,
    prepare_features_for_logistic_regression,
)


# =============================================================================
# REFINED HIERARCHICAL BAYESIAN NETWORK
# =============================================================================

class RefinedHierarchicalBN:
    """
    Refined Bayesian Network with:
    - Better discretization (5-6 bins, percentile-based)
    - Simpler structure (3 layers)
    - Direct connections to outcome
    - Top predictive features only
    """
    
    def __init__(self):
        self.model = None
        self.inference = None
        self.label_encoders = {}
        self.feature_names = []
        
    def _improved_discretization(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Improved discretization strategy:
        - 5-6 bins instead of 3-4
        - Percentile-based for balanced distribution
        - Fit on training, apply to val/test
        """
        train_processed = X_train.copy()
        val_processed = X_val.copy()
        test_processed = X_test.copy()
        
        # Define continuous features to discretize with better bins
        continuous_features = {
            'home_win_pct': 6,  # 6 bins for win percentage
            'away_win_pct': 6,
            'home_ppg': 5,  # 5 bins for scoring
            'away_ppg': 5,
            'strength_differential': 6,  # Key feature - more bins
            'home_last5_win_pct': 5,
            'away_last5_win_pct': 5,
        }
        
        for feature, n_bins in continuous_features.items():
            if feature in train_processed.columns:
                # Use qcut for percentile-based bins (balanced)
                try:
                    train_processed[f'{feature}_binned'], bins = pd.qcut(
                        train_processed[feature], 
                        q=n_bins, 
                        labels=[f'Q{i+1}' for i in range(n_bins)],
                        retbins=True,
                        duplicates='drop'
                    )
                    
                    # Apply same bins to val and test
                    val_processed[f'{feature}_binned'] = pd.cut(
                        val_processed[feature],
                        bins=bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)],
                        include_lowest=True
                    )
                    test_processed[f'{feature}_binned'] = pd.cut(
                        test_processed[feature],
                        bins=bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)],
                        include_lowest=True
                    )
                    
                    # Fill any NaN from out-of-range values
                    train_processed[f'{feature}_binned'] = train_processed[f'{feature}_binned'].fillna('Q3')  # Middle bin
                    val_processed[f'{feature}_binned'] = val_processed[f'{feature}_binned'].fillna('Q3')
                    test_processed[f'{feature}_binned'] = test_processed[f'{feature}_binned'].fillna('Q3')
                    
                except Exception as e:
                    # Fallback to simple binning if qcut fails
                    print(f"  Warning: qcut failed for {feature}, using regular cut")
                    train_processed[f'{feature}_binned'] = pd.cut(
                        train_processed[feature],
                        bins=n_bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)]
                    )
                    val_processed[f'{feature}_binned'] = pd.cut(
                        val_processed[feature],
                        bins=n_bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)]
                    )
                    test_processed[f'{feature}_binned'] = pd.cut(
                        test_processed[feature],
                        bins=n_bins,
                        labels=[f'Q{i+1}' for i in range(n_bins)]
                    )
        
        return train_processed, val_processed, test_processed
    
    def _simplified_dag_structure(self) -> List[Tuple[str, str]]:
        """
        Simplified 3-layer structure with direct connections.
        
        Layer 1 (Roots): home_strength, away_strength, rest_home, rest_away
        Layer 2 (Derived): strength_diff, offensive_matchup, form features
        Layer 3 (Outcome): game_outcome (multiple direct connections)
        """
        edges = [
            # Layer 1 → Layer 2: Basic features create derived features
            ('home_strength', 'strength_differential'),
            ('away_strength', 'strength_differential'),
            ('home_strength', 'offensive_matchup'),
            ('away_strength', 'offensive_matchup'),
            ('home_strength', 'home_form'),
            ('away_strength', 'away_form'),
            ('rest_home', 'rest_advantage'),
            ('rest_away', 'rest_advantage'),
            
            # Layer 2 → Layer 3: Multiple paths to outcome
            ('strength_differential', 'game_outcome'),  # Direct effect
            ('offensive_matchup', 'game_outcome'),      # Direct effect
            ('home_form', 'game_outcome'),              # Direct effect
            ('away_form', 'game_outcome'),              # Direct effect
            ('rest_advantage', 'game_outcome'),         # Direct effect
            
            # Also allow Layer 1 → Layer 3 for key features
            ('home_strength', 'game_outcome'),          # Strong teams win
            ('away_strength', 'game_outcome'),          # Weak opponents help
        ]
        
        return edges
    
    def _extract_top_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer top predictive features.
        Focus on: strength, form, rest, differentials
        """
        data = pd.DataFrame()
        
        # Layer 1: Root nodes (individual team features)
        # Home strength - use binned win_pct if available
        if 'home_win_pct_binned' in X.columns:
            data['home_strength'] = X['home_win_pct_binned'].astype(str)
        elif 'home_team_strength' in X.columns:
            data['home_strength'] = X['home_team_strength'].astype(str)
        else:
            data['home_strength'] = 'Average'
        
        # Away strength
        if 'away_win_pct_binned' in X.columns:
            data['away_strength'] = X['away_win_pct_binned'].astype(str)
        elif 'away_team_strength' in X.columns:
            data['away_strength'] = X['away_team_strength'].astype(str)
        else:
            data['away_strength'] = 'Average'
        
        # Rest features (simplified)
        if 'back_to_back_home' in X.columns:
            data['rest_home'] = X['back_to_back_home'].apply(
                lambda x: 'Tired' if str(x) in ['1', 'True'] else 'Rested'
            )
        else:
            data['rest_home'] = 'Rested'
        
        if 'back_to_back_away' in X.columns:
            data['rest_away'] = X['back_to_back_away'].apply(
                lambda x: 'Tired' if str(x) in ['1', 'True'] else 'Rested'
            )
        else:
            data['rest_away'] = 'Rested'
        
        # Layer 2: Derived features
        # Strength differential (KEY FEATURE - keep high resolution)
        if 'strength_differential_binned' in X.columns:
            data['strength_differential'] = X['strength_differential_binned'].astype(str)
        elif 'strength_differential' in X.columns:
            # Create on the fly if not pre-binned
            data['strength_differential'] = pd.cut(
                X['strength_differential'],
                bins=[-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf],
                labels=['AwayFavored', 'AwaySlightEdge', 'Even', 'HomeSlightEdge', 'HomeFavored']
            ).astype(str)
        else:
            data['strength_differential'] = 'Even'
        
        # Offensive matchup (home offense vs away defense)
        if 'offensive_advantage' in X.columns:
            data['offensive_matchup'] = pd.cut(
                X['offensive_advantage'],
                bins=[-np.inf, -5, 0, 5, np.inf],
                labels=['AwayAdvantage', 'Balanced', 'HomeAdvantage', 'HomeStrong']
            ).astype(str)
        else:
            data['offensive_matchup'] = 'Balanced'
        
        # Recent form
        if 'home_last5_win_pct_binned' in X.columns:
            data['home_form'] = X['home_last5_win_pct_binned'].astype(str)
        elif 'home_recent_form' in X.columns:
            data['home_form'] = X['home_recent_form'].astype(str)
        else:
            data['home_form'] = 'Average'
        
        if 'away_last5_win_pct_binned' in X.columns:
            data['away_form'] = X['away_last5_win_pct_binned'].astype(str)
        elif 'away_recent_form' in X.columns:
            data['away_form'] = X['away_recent_form'].astype(str)
        else:
            data['away_form'] = 'Average'
        
        # Rest advantage
        if 'rest_advantage' in X.columns:
            data['rest_advantage'] = pd.cut(
                X['rest_advantage'],
                bins=[-np.inf, -1, 0, 1, np.inf],
                labels=['AwayRested', 'Same', 'HomeRested', 'HomeVeryRested']
            ).astype(str)
        else:
            # Derive from rest_home and rest_away
            rest_map = {'Tired': 0, 'Rested': 1}
            home_rest = data['rest_home'].map(rest_map)
            away_rest = data['rest_away'].map(rest_map)
            diff = home_rest - away_rest
            data['rest_advantage'] = diff.apply(
                lambda x: 'AwayRested' if x < 0 else ('Same' if x == 0 else 'HomeRested')
            )
        
        return data
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: List[str],
            X_val: pd.DataFrame = None, X_test: pd.DataFrame = None):
        """
        Fit the refined Bayesian Network.
        """
        if not PGMPY_AVAILABLE:
            raise ImportError("pgmpy is required")
        
        print("\nTraining Refined Hierarchical Bayesian Network...")
        print("  Improvements: Better discretization, simplified structure, top features")
        
        self.feature_names = feature_names
        
        # Apply improved discretization
        print("  Applying improved discretization (5-6 bins, percentile-based)...")
        if X_val is not None and X_test is not None:
            X_train_disc, X_val_disc, X_test_disc = self._improved_discretization(
                X_train, X_val, X_test
            )
        else:
            X_train_disc = X_train
        
        # Extract top features
        print("  Extracting top predictive features...")
        dag_data = self._extract_top_features(X_train_disc)
        
        # Add outcome
        dag_data['game_outcome'] = y_train.astype(str)
        
        # Store for later
        self.nodes = list(dag_data.columns)
        
        # Define simplified DAG structure
        print("  Defining simplified DAG structure (3 layers, direct connections)...")
        edges = self._simplified_dag_structure()
        
        print(f"  Network: {len(self.nodes)} nodes, {len(edges)} edges")
        
        # Create Bayesian Network
        print("  Creating Bayesian Network...")
        self.model = BayesianNetwork(edges)
        
        # Learn parameters with Bayesian estimation (better than MLE for sparse data)
        print("  Learning CPDs with Bayesian estimation (Dirichlet prior)...")
        try:
            self.model.fit(
                dag_data,
                estimator=BayesianEstimator,
                prior_type='BDeu',  # Bayesian Dirichlet equivalent uniform
                equivalent_sample_size=10  # Smoothing parameter
            )
        except:
            # Fallback to MLE if Bayesian fails
            print("  Bayesian estimation failed, using MLE...")
            self.model.fit(dag_data, estimator=MaximumLikelihoodEstimator)
        
        # Verify model
        if self.model.check_model():
            print("  ✓ Model structure is valid")
        else:
            print("  ✗ Warning: Model validation failed")
        
        # Create inference
        print("  Setting up inference engine...")
        self.inference = VariableElimination(self.model)
        
        print("  Training complete!")
        
        # Store discretized val/test for prediction
        if X_val is not None and X_test is not None:
            self.X_val_disc = X_val_disc
            self.X_test_disc = X_test_disc
    
    def predict(self, X: pd.DataFrame, use_stored_disc: bool = False) -> np.ndarray:
        """Predict class labels."""
        if self.inference is None:
            raise ValueError("Model not trained yet!")
        
        # Use stored discretized version if available (for val/test)
        if use_stored_disc and hasattr(self, 'X_val_disc'):
            dag_data = self._extract_top_features(self.X_val_disc if 'val' in str(X.shape) else self.X_test_disc)
        else:
            dag_data = self._extract_top_features(X)
        
        predictions = []
        
        for idx in range(len(dag_data)):
            evidence = dag_data.iloc[idx].to_dict()
            
            try:
                result = self.inference.query(
                    variables=['game_outcome'],
                    evidence=evidence
                )
                probs = result.values
                pred = np.argmax(probs)
                predictions.append(pred)
            except:
                predictions.append(1)  # Default to home win
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame, use_stored_disc: bool = False) -> np.ndarray:
        """Predict class probabilities."""
        if self.inference is None:
            raise ValueError("Model not trained yet!")
        
        if use_stored_disc and hasattr(self, 'X_val_disc'):
            dag_data = self._extract_top_features(self.X_val_disc if 'val' in str(X.shape) else self.X_test_disc)
        else:
            dag_data = self._extract_top_features(X)
        
        probabilities = []
        
        for idx in range(len(dag_data)):
            evidence = dag_data.iloc[idx].to_dict()
            
            try:
                result = self.inference.query(
                    variables=['game_outcome'],
                    evidence=evidence
                )
                probs = result.values
                if len(probs) == 2:
                    probabilities.append(probs)
                else:
                    probabilities.append([0.5, 0.5])
            except:
                probabilities.append([0.44, 0.56])
        
        return np.array(probabilities)
    
    def get_network_structure(self) -> str:
        """Return network structure."""
        if self.model is None:
            return "Model not trained"
        
        structure = "\nRefined Hierarchical BN Structure:\n"
        structure += "="*60 + "\n"
        structure += f"Nodes: {len(self.nodes)}\n"
        structure += f"Edges: {len(self.model.edges())}\n"
        structure += "\nKey improvements:\n"
        structure += "  • 5-6 bins (vs 3-4)\n"
        structure += "  • 3 layers (vs 5)\n"
        structure += "  • Direct connections to outcome\n"
        structure += "  • Top predictive features only\n"
        structure += "  • Bayesian estimation with priors\n"
        
        return structure


def prepare_features_for_refined_dag(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for refined DAG.
    Include continuous features that will be discretized internally.
    """
    # Core features we need
    core_features = [
        # Win percentages (for strength)
        'home_win_pct', 'away_win_pct',
        # PPG (for offense)
        'home_ppg', 'away_ppg', 'home_opp_ppg', 'away_opp_ppg',
        # Recent form
        'home_last5_win_pct', 'away_last5_win_pct',
        # Rest
        'back_to_back_home', 'back_to_back_away',
        # Derived features
        'strength_differential', 'offensive_advantage', 'defensive_advantage',
        'rest_advantage',
        # Categorical if available
        'home_team_strength', 'away_team_strength',
        'home_recent_form', 'away_recent_form',
    ]
    
    # Filter to existing columns
    feature_cols = [col for col in core_features if col in train.columns]
    
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Handle missing
    for col in feature_cols:
        if X_train[col].dtype in ['float64', 'int64']:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_val[col] = X_val[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        else:
            # For categorical columns, add 'Missing' to categories first
            if X_train[col].dtype.name == 'category':
                if 'Missing' not in X_train[col].cat.categories:
                    X_train[col] = X_train[col].cat.add_categories(['Missing'])
                if 'Missing' not in X_val[col].cat.categories:
                    X_val[col] = X_val[col].cat.add_categories(['Missing'])
                if 'Missing' not in X_test[col].cat.categories:
                    X_test[col] = X_test[col].cat.add_categories(['Missing'])
            
            X_train[col] = X_train[col].fillna('Missing').astype(str)
            X_val[col] = X_val[col].fillna('Missing').astype(str)
            X_test[col] = X_test[col].fillna('Missing').astype(str)
    
    y_train = train['game_outcome'].values
    y_val = val['game_outcome'].values
    y_test = test['game_outcome'].values
    
    print(f"\nRefined BN Features:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main(force_reprocess: bool = False):
    """Main training pipeline for refined DAG."""
    print("="*80)
    print("REFINED HIERARCHICAL BAYESIAN NETWORK")
    print("="*80)
    print("\nImprovements:")
    print("  1. Better discretization (5-6 bins, percentile-based)")
    print("  2. Simplified structure (3 layers, direct connections)")
    print("  3. Key features (strength_differential, interaction terms)")
    print("  4. Feature selection (top predictive features)")
    print("  5. Bayesian estimation (vs MLE)")
    
    # 1. Load data
    print("\n[1/4] Loading processed data...")
    df = load_or_create_processed_data(force_reprocess=force_reprocess)
    
    # 2. Split
    print("\n[2/4] Splitting data...")
    train, val, test = split_train_val_test(df, TRAIN_END_SEASON, VAL_END_SEASON)
    
    # 3. Train LR baseline
    print("\n[3/4] Training Logistic Regression baseline...")
    X_train_lr, X_val_lr, X_test_lr, y_train, y_val, y_test, feature_names_lr = \
        prepare_features_for_logistic_regression(train, val, test)
    
    lr_model = LogisticRegressionBaseline()
    lr_model.fit(X_train_lr, y_train, feature_names_lr)
    
    y_pred_lr = lr_model.predict(X_test_lr)
    y_proba_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    
    print(f"\nLR Test: Accuracy={accuracy_score(y_test, y_pred_lr):.4f}, ROC AUC={roc_auc_score(y_test, y_proba_lr):.4f}")
    
    # 4. Train Refined DAG
    print("\n[4/4] Training Refined Hierarchical BN...")
    
    if not PGMPY_AVAILABLE:
        print("ERROR: pgmpy not installed!")
        return {'lr_model': lr_model, 'dag_model': None}
    
    X_train_dag, X_val_dag, X_test_dag, y_train_dag, y_val_dag, y_test_dag, feature_names_dag = \
        prepare_features_for_refined_dag(train, val, test)
    
    dag_model = RefinedHierarchicalBN()
    dag_model.fit(X_train_dag, y_train_dag, feature_names_dag, X_val_dag, X_test_dag)
    
    print(dag_model.get_network_structure())
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred_dag = dag_model.predict(X_test_dag)
    y_proba_dag = dag_model.predict_proba(X_test_dag)[:, 1]
    
    print(f"\nRefined DAG Test: Accuracy={accuracy_score(y_test_dag, y_pred_dag):.4f}, ROC AUC={roc_auc_score(y_test_dag, y_proba_dag):.4f}")
    
    # Comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    print(f"{'Model':<35} {'Accuracy':<12} {'ROC AUC':<12}")
    print("-"*80)
    print(f"{'Logistic Regression':<35} {accuracy_score(y_test, y_pred_lr):<12.4f} {roc_auc_score(y_test, y_proba_lr):<12.4f}")
    print(f"{'Refined Hierarchical BN':<35} {accuracy_score(y_test_dag, y_pred_dag):<12.4f} {roc_auc_score(y_test_dag, y_proba_dag):<12.4f}")
    
    improvement = accuracy_score(y_test_dag, y_pred_dag) - accuracy_score(y_test, y_pred_lr)
    print(f"\nImprovement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return {
        'lr_model': lr_model,
        'dag_model': dag_model,
        'results': {
            'lr': {
                'accuracy': accuracy_score(y_test, y_pred_lr),
                'roc_auc': roc_auc_score(y_test, y_proba_lr),
            },
            'dag': {
                'accuracy': accuracy_score(y_test_dag, y_pred_dag),
                'roc_auc': roc_auc_score(y_test_dag, y_proba_dag),
            }
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train refined hierarchical BN")
    parser.add_argument("--reprocess", action="store_true")
    
    args = parser.parse_args()
    results = main(force_reprocess=args.reprocess)

