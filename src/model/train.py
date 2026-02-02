"""
Training and evaluation for word duration prediction models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from .model import WordDurationPredictor, SimpleDurationPredictor
from .dataset import create_dataloader


def train_model(
    data_path: str,
    output_dir: str,
    model_type: str = "simple",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "auto",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Train a word duration prediction model.

    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save model
        model_type: 'transformer' or 'simple'
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on ('auto', 'cuda', 'cpu')
        verbose: Print progress

    Returns:
        Dictionary with training metrics
    """
    # Device setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"Training on {device}")
        print(f"Model type: {model_type}")

    # Create dataloaders
    train_loader = create_dataloader(
        data_path, batch_size=batch_size, split="train", model_type=model_type
    )
    val_loader = create_dataloader(
        data_path, batch_size=batch_size, split="val", model_type=model_type
    )

    if verbose:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    if model_type == "transformer":
        model = WordDurationPredictor()
        learning_rate = 2e-5  # Lower LR for transformers
    else:
        model = SimpleDurationPredictor()
        # Copy vocab from dataset
        if hasattr(train_loader.dataset, "word2idx"):
            model.word2idx = train_loader.dataset.word2idx

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader

        for batch in iterator:
            optimizer.zero_grad()

            if model_type == "transformer":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                durations = batch["durations"].to(device)
                mask = batch["word_mask"].to(device)

                predictions = model(input_ids, attention_mask)
            else:
                word_ids = batch["word_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                lengths = batch["lengths"].to(device)
                mask = batch["mask"].to(device).float()
                durations = batch["durations"].to(device)

                predictions = model(word_ids, char_ids, lengths, mask.bool())

            # Masked loss
            loss = criterion(predictions, durations)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        val_loss, val_mae = evaluate_model(model, val_loader, device, model_type)

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if verbose:
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(output_dir)
            if verbose:
                print(f"  Saved best model (val_loss: {val_loss:.4f})")

    # Save training history
    with open(Path(output_dir) / "history.json", "w") as f:
        json.dump(history, f)

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": history["train_loss"][-1],
        "final_val_mae": history["val_mae"][-1],
    }


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    model_type: str = "simple"
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device for evaluation
        model_type: 'transformer' or 'simple'

    Returns:
        Tuple of (loss, mean_absolute_error)
    """
    model.eval()
    criterion = nn.SmoothL1Loss(reduction="none")

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_type == "transformer":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                durations = batch["durations"].to(device)
                mask = batch["word_mask"].to(device)

                predictions = model(input_ids, attention_mask)
            else:
                word_ids = batch["word_ids"].to(device)
                char_ids = batch["char_ids"].to(device)
                lengths = batch["lengths"].to(device)
                mask = batch["mask"].to(device).float()
                durations = batch["durations"].to(device)

                predictions = model(word_ids, char_ids, lengths, mask.bool())

            # Masked metrics
            loss = criterion(predictions, durations)
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)

            mae = torch.abs(predictions - durations)
            mae = (mae * mask).sum() / (mask.sum() + 1e-8)

            batch_samples = mask.sum().item()
            total_loss += loss.item() * batch_samples
            total_mae += mae.item() * batch_samples
            total_samples += batch_samples

    avg_loss = total_loss / (total_samples + 1e-8)
    avg_mae = total_mae / (total_samples + 1e-8)

    return avg_loss, avg_mae


def quick_train(
    data_path: str,
    output_dir: str = "./models/duration_predictor",
    epochs: int = 10
) -> Dict[str, float]:
    """
    Quick training function with sensible defaults.

    Args:
        data_path: Path to training CSV
        output_dir: Where to save model
        epochs: Number of epochs

    Returns:
        Training metrics
    """
    return train_model(
        data_path=data_path,
        output_dir=output_dir,
        model_type="simple",
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-3,
        device="auto",
        verbose=True
    )
