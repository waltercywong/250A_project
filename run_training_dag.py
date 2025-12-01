#!/usr/bin/env python3
"""
Quick script to run the hierarchical DAG Bayesian Network training pipeline.

This trains a structured Bayesian Network with proper dependencies (NOT Naive Bayes).

Usage:
    python run_training_dag.py              # Use existing processed data
    python run_training_dag.py --reprocess  # Force reprocessing of data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.train_models_dag import main

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train hierarchical DAG Bayesian Network for NBA game prediction"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data even if processed file exists"
    )
    
    args = parser.parse_args()
    
    # Check if pgmpy is installed
    try:
        import pgmpy
        print("✓ pgmpy is installed")
    except ImportError:
        print("\n" + "="*80)
        print("ERROR: pgmpy not installed!")
        print("="*80)
        print("\nThe hierarchical Bayesian Network requires pgmpy.")
        print("\nInstall it with:")
        print("  pip install pgmpy")
        print("\nOr run the installation script:")
        print("  ./install_dag_requirements.sh")
        print("\n" + "="*80)
        sys.exit(1)
    
    # Run training
    print("\n" + "="*80)
    print("HIERARCHICAL DAG BAYESIAN NETWORK TRAINING")
    print("="*80)
    print("\nThis trains a structured Bayesian Network with proper dependencies.")
    print("NOT Naive Bayes - features influence each other in a hierarchy!")
    print()
    
    try:
        results = main(force_reprocess=args.reprocess)
        
        print("\n✅ Training complete! Results:")
        if results.get('dag_model'):
            print("\n📊 Performance Summary:")
            print(f"  Logistic Regression: {results['results']['lr']['accuracy']:.1%} accuracy")
            print(f"  Hierarchical DAG:    {results['results']['dag']['accuracy']:.1%} accuracy")
        
        print("\n📁 Check the 'results' folder for detailed outputs.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

