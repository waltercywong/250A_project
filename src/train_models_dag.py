"""
Training script for structured Bayesian Network with hierarchical DAG.

This implements the network structure:
    Game_History (root) 
        → Home_Strength, Away_Strength
            → Home_Off/Def_Strength, Away_Off/Def_Strength
                → Home/Away_Recent_Form
                    → Matchup_Advantage ← Days_Rest_Home/Away
                        → Home_Court_Factor
                            → Game_Outcome

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

# Try pgmpy for Bayesian Networks (better for structure learning)
try:
    from pgmpy.models import BayesianNetwork as BayesianNetworkOld
    # Use DiscreteBayesianNetwork for newer versions
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        # Fallback to old API if available
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
# HIERARCHICAL BAYESIAN NETWORK WITH DAG STRUCTURE
# =============================================================================

class HierarchicalBayesianNetwork:
    """
    Bayesian Network with hierarchical DAG structure (NOT Naive Bayes).
    
    Network Structure (as per project plan):
        Game_History (always 1 - represents we have historical data)
            ↓         ↓
        Home_Strength  Away_Strength
            ↓              ↓
        Home_Off_Strength  Away_Off_Strength
        Home_Def_Strength  Away_Def_Strength
            ↓              ↓
        Home_Recent_Form   Away_Recent_Form
            ↓              ↓
        Days_Rest_Home → Matchup_Advantage ← Days_Rest_Away
                           ↓
                    Home_Court_Factor
                           ↓
                      Game_Outcome
    
    Key differences from Naive Bayes:
    - Features are NOT conditionally independent
    - Hierarchical structure: strength → offense/defense → form → matchup → outcome
    - Rest days directly influence matchup advantage
    """
    
    def __init__(self):
        self.model = None
        self.inference = None
        self.label_encoders = {}
        self.feature_names = []
        self.node_mapping = {}  # Maps our feature names to DAG nodes
        
    def _define_dag_structure(self) -> List[Tuple[str, str]]:
        """
        Define the DAG structure as edges (parent, child).
        
        Returns:
            List of (parent, child) tuples defining the network structure
        """
        edges = [
            # Team strength influences offensive/defensive capabilities
            ('home_strength', 'home_off_strength'),
            ('home_strength', 'home_def_strength'),
            ('away_strength', 'away_off_strength'),
            ('away_strength', 'away_def_strength'),
            
            # Offensive/defensive strength influences recent form
            ('home_off_strength', 'home_recent_form'),
            ('home_def_strength', 'home_recent_form'),
            ('away_off_strength', 'away_recent_form'),
            ('away_def_strength', 'away_recent_form'),
            
            # Recent form and rest influence matchup advantage
            ('home_recent_form', 'matchup_advantage'),
            ('away_recent_form', 'matchup_advantage'),
            ('days_rest_home', 'matchup_advantage'),
            ('days_rest_away', 'matchup_advantage'),
            
            # Matchup advantage influences home court factor
            ('matchup_advantage', 'home_court_factor'),
            
            # Home court factor determines outcome
            ('home_court_factor', 'game_outcome'),
        ]
        
        return edges
    
    def _map_features_to_nodes(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw features to DAG nodes.
        
        We need to map our engineered features to the conceptual nodes in the DAG.
        """
        data = pd.DataFrame()
        
        # Home strength (use categorized team strength)
        if 'home_team_strength' in X_train.columns:
            data['home_strength'] = X_train['home_team_strength'].astype(str)
        else:
            # Fallback: create from win_pct if available
            data['home_strength'] = 'Average'
        
        # Away strength
        if 'away_team_strength' in X_train.columns:
            data['away_strength'] = X_train['away_team_strength'].astype(str)
        else:
            data['away_strength'] = 'Average'
        
        # Home offensive/defensive strength
        if 'home_offensive_strength' in X_train.columns:
            data['home_off_strength'] = X_train['home_offensive_strength'].astype(str)
        else:
            data['home_off_strength'] = 'Average'
            
        if 'home_defensive_strength' in X_train.columns:
            data['home_def_strength'] = X_train['home_defensive_strength'].astype(str)
        else:
            data['home_def_strength'] = 'Average'
        
        # Away offensive/defensive strength
        if 'away_offensive_strength' in X_train.columns:
            data['away_off_strength'] = X_train['away_offensive_strength'].astype(str)
        else:
            data['away_off_strength'] = 'Average'
            
        if 'away_defensive_strength' in X_train.columns:
            data['away_def_strength'] = X_train['away_defensive_strength'].astype(str)
        else:
            data['away_def_strength'] = 'Average'
        
        # Recent form
        if 'home_recent_form' in X_train.columns:
            data['home_recent_form'] = X_train['home_recent_form'].astype(str)
        else:
            data['home_recent_form'] = 'Average'
            
        if 'away_recent_form' in X_train.columns:
            data['away_recent_form'] = X_train['away_recent_form'].astype(str)
        else:
            data['away_recent_form'] = 'Average'
        
        # Rest days (categorized)
        if 'days_rest_home_cat' in X_train.columns:
            data['days_rest_home'] = X_train['days_rest_home_cat'].astype(str)
        elif 'back_to_back_home' in X_train.columns:
            data['days_rest_home'] = X_train['back_to_back_home'].apply(
                lambda x: 'B2B' if str(x) == '1' or str(x) == 'True' else 'Rested'
            )
        else:
            data['days_rest_home'] = 'Rested'
            
        if 'days_rest_away_cat' in X_train.columns:
            data['days_rest_away'] = X_train['days_rest_away_cat'].astype(str)
        elif 'back_to_back_away' in X_train.columns:
            data['days_rest_away'] = X_train['back_to_back_away'].apply(
                lambda x: 'B2B' if str(x) == '1' or str(x) == 'True' else 'Rested'
            )
        else:
            data['days_rest_away'] = 'Rested'
        
        # Matchup advantage
        if 'matchup_advantage' in X_train.columns:
            data['matchup_advantage'] = X_train['matchup_advantage'].astype(str)
        else:
            # Derive from strength differential
            data['matchup_advantage'] = 'Even'
        
        # Home court factor
        if 'home_court_advantage' in X_train.columns:
            data['home_court_factor'] = X_train['home_court_advantage'].apply(
                lambda x: 'Home' if str(x) == '1' else 'Neutral'
            )
        else:
            data['home_court_factor'] = 'Home'
        
        return data
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_names: List[str]):
        """
        Fit the hierarchical Bayesian Network.
        
        Args:
            X_train: Feature dataframe (categorical/discrete)
            y_train: Target array
            feature_names: List of feature column names (for reference)
        """
        if not PGMPY_AVAILABLE:
            raise ImportError(
                "pgmpy is required for structured Bayesian Networks. "
                "Install with: pip install pgmpy"
            )
        
        print("\nTraining Hierarchical Bayesian Network with DAG structure...")
        print("  (This is NOT Naive Bayes - features have dependencies)")
        
        self.feature_names = feature_names
        
        # Map features to DAG nodes
        print("  Mapping features to DAG nodes...")
        dag_data = self._map_features_to_nodes(X_train)
        
        # Add outcome
        dag_data['game_outcome'] = y_train.astype(str)
        
        # Store node names
        self.nodes = list(dag_data.columns)
        
        # Define DAG structure
        print("  Defining DAG structure...")
        edges = self._define_dag_structure()
        
        # Create Bayesian Network
        print("  Creating Bayesian Network...")
        self.model = BayesianNetwork(edges)
        
        # Learn parameters (CPDs) from data
        print("  Learning conditional probability distributions...")
        self.model.fit(
            dag_data,
            estimator=MaximumLikelihoodEstimator
        )
        
        # Verify model
        if self.model.check_model():
            print("  ✓ Model structure is valid (acyclic)")
        else:
            print("  ✗ Warning: Model may have issues")
        
        # Create inference object
        print("  Setting up inference engine...")
        self.inference = VariableElimination(self.model)
        
        print("  Training complete")
        print(f"  Network nodes: {len(self.nodes)}")
        print(f"  Network edges: {len(edges)}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.inference is None:
            raise ValueError("Model not trained yet!")
        
        # Map features to DAG nodes
        dag_data = self._map_features_to_nodes(X)
        
        predictions = []
        
        for idx in range(len(dag_data)):
            # Get evidence for this instance
            evidence = dag_data.iloc[idx].to_dict()
            
            # Query for game outcome
            try:
                result = self.inference.query(
                    variables=['game_outcome'],
                    evidence=evidence
                )
                
                # Get most likely outcome
                probs = result.values
                pred = np.argmax(probs)
                predictions.append(pred)
            except Exception as e:
                # Fallback: predict most common class
                predictions.append(1)  # Home win
        
        return np.array(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.inference is None:
            raise ValueError("Model not trained yet!")
        
        # Map features to DAG nodes
        dag_data = self._map_features_to_nodes(X)
        
        probabilities = []
        
        for idx in range(len(dag_data)):
            # Get evidence for this instance
            evidence = dag_data.iloc[idx].to_dict()
            
            # Query for game outcome
            try:
                result = self.inference.query(
                    variables=['game_outcome'],
                    evidence=evidence
                )
                
                # Get probability distribution
                probs = result.values
                # Ensure it's [P(0), P(1)]
                if len(probs) == 2:
                    probabilities.append(probs)
                else:
                    probabilities.append([0.5, 0.5])
            except Exception as e:
                # Fallback
                probabilities.append([0.44, 0.56])  # Slight home advantage
        
        return np.array(probabilities)
    
    def get_network_structure(self) -> str:
        """Return a string representation of the network structure."""
        if self.model is None:
            return "Model not trained yet"
        
        structure = "\nHierarchical Bayesian Network Structure:\n"
        structure += "="*60 + "\n"
        
        # Get edges
        edges = self.model.edges()
        
        # Group by parent
        from collections import defaultdict
        children = defaultdict(list)
        for parent, child in edges:
            children[parent].append(child)
        
        # Print hierarchy
        def print_node(node, indent=0):
            result = "  " * indent + f"• {node}\n"
            if node in children:
                for child in sorted(children[node]):
                    result += print_node(child, indent + 1)
            return result
        
        # Find root nodes (nodes with no parents)
        all_children = set()
        for parent, child in edges:
            all_children.add(child)
        
        all_nodes = set([parent for parent, _ in edges] + [child for _, child in edges])
        root_nodes = all_nodes - all_children
        
        # Print from each root
        for root in sorted(root_nodes):
            structure += print_node(root)
        
        return structure


def prepare_features_for_dag(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for the hierarchical Bayesian Network.
    We need the discretized/categorical features.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    # Select categorical features needed for DAG
    categorical_features = [
        'home_team_strength', 'home_offensive_strength', 'home_defensive_strength',
        'away_team_strength', 'away_offensive_strength', 'away_defensive_strength',
        'home_recent_form', 'away_recent_form',
        'matchup_advantage', 'season_stage',
    ]
    
    discrete_features = [
        'back_to_back_home', 'back_to_back_away',
        'home_court_advantage', 'is_weekend',
    ]
    
    rest_features = ['days_rest_home_cat', 'days_rest_away_cat']
    
    feature_cols = categorical_features + discrete_features + rest_features
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in train.columns]
    
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Convert all to string for categorical handling
    for col in feature_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # Handle missing values
    X_train = X_train.fillna('Missing')
    X_val = X_val.fillna('Missing')
    X_test = X_test.fillna('Missing')
    
    y_train = train['game_outcome'].values
    y_val = val['game_outcome'].values
    y_test = test['game_outcome'].values
    
    print(f"\nHierarchical BN Features:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Val samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main(force_reprocess: bool = False):
    """Main training pipeline with hierarchical Bayesian Network."""
    print("="*80)
    print("NBA GAME OUTCOME PREDICTION - HIERARCHICAL BAYESIAN NETWORK")
    print("="*80)
    
    # 1. Load/create processed data
    print("\n[1/4] Loading processed data...")
    df = load_or_create_processed_data(force_reprocess=force_reprocess)
    
    # 2. Split data
    print("\n[2/4] Splitting data into train/val/test...")
    train, val, test = split_train_val_test(df, TRAIN_END_SEASON, VAL_END_SEASON)
    
    # 3. Train Logistic Regression (baseline)
    print("\n[3/4] Training Logistic Regression baseline...")
    X_train_lr, X_val_lr, X_test_lr, y_train, y_val, y_test, feature_names_lr = \
        prepare_features_for_logistic_regression(train, val, test)
    
    lr_model = LogisticRegressionBaseline()
    lr_model.fit(X_train_lr, y_train, feature_names_lr)
    
    # Evaluate LR
    y_pred_lr = lr_model.predict(X_test_lr)
    y_proba_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    
    print(f"\nLogistic Regression Test Results:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"  ROC AUC:  {roc_auc_score(y_test, y_proba_lr):.4f}")
    
    # 4. Train Hierarchical Bayesian Network
    print("\n[4/4] Training Hierarchical Bayesian Network (DAG structure)...")
    
    if not PGMPY_AVAILABLE:
        print("\n" + "="*80)
        print("ERROR: pgmpy not installed!")
        print("="*80)
        print("\nTo train the hierarchical Bayesian Network, install pgmpy:")
        print("  pip install pgmpy")
        print("\nFor now, only Logistic Regression baseline is available.")
        return {
            'lr_model': lr_model,
            'dag_model': None,
        }
    
    X_train_dag, X_val_dag, X_test_dag, y_train_dag, y_val_dag, y_test_dag, feature_names_dag = \
        prepare_features_for_dag(train, val, test)
    
    dag_model = HierarchicalBayesianNetwork()
    dag_model.fit(X_train_dag, y_train_dag, feature_names_dag)
    
    # Print network structure
    print(dag_model.get_network_structure())
    
    # Evaluate DAG model
    print("\nEvaluating Hierarchical BN on test set...")
    y_pred_dag = dag_model.predict(X_test_dag)
    y_proba_dag = dag_model.predict_proba(X_test_dag)[:, 1]
    
    print(f"\nHierarchical Bayesian Network Test Results:")
    print(f"  Accuracy: {accuracy_score(y_test_dag, y_pred_dag):.4f}")
    print(f"  ROC AUC:  {roc_auc_score(y_test_dag, y_proba_dag):.4f}")
    
    # Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'ROC AUC':<12}")
    print("-"*80)
    print(f"{'Logistic Regression':<30} {accuracy_score(y_test, y_pred_lr):<12.4f} {roc_auc_score(y_test, y_proba_lr):<12.4f}")
    print(f"{'Hierarchical Bayesian Net':<30} {accuracy_score(y_test_dag, y_pred_dag):<12.4f} {roc_auc_score(y_test_dag, y_proba_dag):<12.4f}")
    
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
    
    parser = argparse.ArgumentParser(description="Train hierarchical Bayesian Network")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data"
    )
    
    args = parser.parse_args()
    
    results = main(force_reprocess=args.reprocess)

