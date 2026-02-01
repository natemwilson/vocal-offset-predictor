#!/usr/bin/env python3
"""
Train a word duration prediction model.

Usage:
    python train.py --data data.csv --output models/my_model
    python train.py --data data.csv --epochs 20 --model-type transformer
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train a word duration prediction model"
    )

    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to training data CSV"
    )

    parser.add_argument(
        "--output", "-o",
        default="models/duration_predictor",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--model-type", "-m",
        choices=["simple", "transformer"],
        default="simple",
        help="Model architecture to use"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for training"
    )

    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Device to train on (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Validate data path
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("Vocal Offset Predictor - Training")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Train model
    metrics = train_model(
        data_path=str(data_path),
        output_dir=args.output,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"Final training loss: {metrics['final_train_loss']:.4f}")
    print(f"Final validation MAE: {metrics['final_val_mae']:.4f}s")
    print(f"\nModel saved to: {args.output}")


if __name__ == "__main__":
    main()
