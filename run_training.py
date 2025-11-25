#!/usr/bin/env python3
"""
Quick script to run the training pipeline.

Usage:
    python run_training.py              # Use existing processed data
    python run_training.py --reprocess  # Force reprocessing of data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.train_models import main

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NBA game outcome prediction models")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data even if processed file exists"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = main(force_reprocess=args.reprocess)
    
    print("\n✅ Training complete! Check the 'results' folder for outputs.")

