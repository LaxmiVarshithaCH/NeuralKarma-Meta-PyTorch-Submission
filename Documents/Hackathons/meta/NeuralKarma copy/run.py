#!/usr/bin/env python3
"""
NeuralKarma — Single Entry Point
Downloads datasets, trains models, and starts the server.

Usage:
    python run.py              # Full pipeline: download → train → serve
    python run.py --setup      # Only download datasets and train models
    python run.py --serve      # Only start the server (models must exist)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_setup():
    """Download datasets and train ML models."""
    print("\n" + "═" * 60)
    print("  NeuralKarma — Setup Pipeline")
    print("═" * 60 + "\n")

    # Step 1: Download datasets
    print("Phase 1: Downloading real datasets from HuggingFace...\n")
    from data.download_datasets import download_all
    prosocial_df, ethics_data, norms_df = download_all()

    # Step 2: Train models
    print("\nPhase 2: Training ML classifiers...\n")
    from ml.train_models import train_all_models
    models = train_all_models()

    print("\n" + "═" * 60)
    print("  [OK] Setup complete! Run `python run.py --serve` to start.")
    print("═" * 60 + "\n")

    return models


def run_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    import uvicorn

    print("\n" + "═" * 60)
    print("  NeuralKarma — Starting Server")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  API Docs:  http://localhost:{port}/docs")
    print("═" * 60 + "\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def check_models_exist():
    """Check if trained models exist."""
    model_dir = PROJECT_ROOT / "ml" / "models"
    if not model_dir.exists():
        return False
    required = ["prosociality_model.joblib", "prosociality_vectorizer.joblib"]
    return all((model_dir / f).exists() for f in required)


def main():
    parser = argparse.ArgumentParser(
        description="NeuralKarma — AI-Powered Ethical Impact Scoring Engine"
    )
    parser.add_argument("--setup", action="store_true",
                        help="Only download datasets and train models")
    parser.add_argument("--serve", action="store_true",
                        help="Only start the server (assumes models exist)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    args = parser.parse_args()

    if args.setup:
        run_setup()
        return

    if args.serve:
        if not check_models_exist():
            print("[WARNING] Models not found! Running setup first...")
            run_setup()
        run_server(args.host, args.port)
        return

    # Default: full pipeline
    if not check_models_exist():
        run_setup()
    else:
        print("[OK] Models found, skipping setup (use --setup to force)")

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
