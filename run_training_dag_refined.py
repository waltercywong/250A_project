#!/usr/bin/env python3
"""
Run script for Refined Hierarchical DAG Bayesian Network.

Improvements over basic DAG:
1. Better discretization (5-6 bins, percentile-based)
2. Simplified structure (3 layers vs 5)
3. Key features (strength_differential, interactions)
4. Feature selection (top predictive features only)
5. Bayesian estimation with priors

Usage:
    python run_training_dag_refined.py
    python run_training_dag_refined.py --reprocess
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.train_models_dag_refined import main

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train refined hierarchical DAG Bayesian Network"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data"
    )
    
    args = parser.parse_args()
    
    # Check pgmpy
    try:
        import pgmpy
        print("✓ pgmpy is installed")
    except ImportError:
        print("\n" + "="*80)
        print("ERROR: pgmpy not installed!")
        print("="*80)
        print("\nInstall with: pip install pgmpy")
        print("="*80)
        sys.exit(1)
    
    # Run training
    print("\n" + "="*80)
    print("REFINED HIERARCHICAL DAG BAYESIAN NETWORK")
    print("="*80)
    print("\n🎯 IMPROVEMENTS OVER BASIC DAG:")
    print("  ✓ Better discretization (5-6 bins, percentile-based)")
    print("  ✓ Simplified structure (3 layers, direct connections)")
    print("  ✓ Key features (strength_differential included)")
    print("  ✓ Feature selection (top predictors only)")
    print("  ✓ Bayesian estimation (Dirichlet priors)")
    print()
    
    try:
        results = main(force_reprocess=args.reprocess)
        
        if results.get('dag_model'):
            print("\n✅ Training complete!")
            print("\n📊 Final Results:")
            print(f"  Logistic Regression: {results['results']['lr']['accuracy']:.1%} accuracy, {results['results']['lr']['roc_auc']:.3f} ROC AUC")
            print(f"  Refined DAG BN:      {results['results']['dag']['accuracy']:.1%} accuracy, {results['results']['dag']['roc_auc']:.3f} ROC AUC")
            
            improvement = results['results']['dag']['accuracy'] - results['results']['lr']['accuracy']
            if improvement > 0:
                print(f"\n  🎉 Improvement: +{improvement:.1%} vs LR baseline!")
            elif improvement > -0.01:
                print(f"\n  ✓ Comparable to LR (within 1%)")
            else:
                print(f"\n  📉 Still {abs(improvement):.1%} below LR")
                print("     But provides superior interpretability!")
                print("\n  💡 Why the gap?")
                print("     • Information loss from discretization (5-6 bins vs continuous)")
                print("     • Complex inference with limited data")
                print("     • Trade-off: accuracy vs interpretability")
                print("\n  🎯 Value of DAG:")
                print("     • Can trace causal paths (why predictions are made)")
                print("     • Probabilistic queries ('what if' scenarios)")
                print("     • Encodes domain knowledge (basketball structure)")
                print("\n  💡 See REFINED_DAG_SUMMARY.md for full analysis")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

