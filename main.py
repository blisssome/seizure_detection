"""
Main script for training and evaluating a Transformer-based EEG seizure classification model.

This script supports both training and evaluation modes:
- When `--train` is passed, the model is trained and the best checkpoint is saved.
- When `--model` is passed, a different model architecture can be selected.

Workflow:
1. Load raw EEG and label data from disk.
2. Downsample EEG signals.
3. Prepare datasets and PyTorch dataloaders.
4. Dynamically initialize a selected model from the model registry.
5. Optionally train and checkpoint the model.
6. Load the best model checkpoint.
7. Evaluate the model on validation data.
8. Print a classification report and save the confusion matrix.

Modules used:
- `loader.py`: for loading and downsampling EEG data.
- `dataset.py`: defines EEGDataset and data loader creation.
- `models/registry.py`: for selecting models dynamically.
- `utils.py`: training, evaluation, plotting, and reporting utilities.
- `checkpoint_manager.py`: saving/loading model checkpoints.

Usage:
    python main.py                # Evaluate using the default Transformer model
    python main.py --train        # Train the default model
    python main.py --model cnn    # Evaluate using the CNN model
    python main.py --train --model rnn  # Train and evaluate using the RNN model
"""

import argparse
from loader import read_data, downsample_eeg
from dataset import EEGDataset
from models.registry import get_model
from checkpoint_manager import CheckpointManager
from utils import (train_model_pipeline, 
                   evaluate_model,
                   get_device)


def main(train=False, model_name="transformer"):
    device = get_device()

    # Load and preprocess data
    train_eeg, train_label, val_eeg, val_label = read_data()
    print("Loaded data.")

    train_eeg_clean = downsample_eeg(train_eeg)
    val_eeg_clean = downsample_eeg(val_eeg)
    print("Downsampled EEG.")
    
    n_channels = train_eeg.shape[1]
    train_dataset = EEGDataset(train_eeg_clean, train_label)
    val_dataset = EEGDataset(val_eeg_clean, val_label)

    train_loader = train_dataset.create_loader()
    val_loader = val_dataset.create_loader()
    print("Created Loaders.")

    # Initialize model dynamically
    model = get_model(name=model_name, n_channels=n_channels).to(device)

    checkpoint = CheckpointManager(checkpoint_dir="checkpoints", best_model_name=f"{model_name}_best.pt")

    if train or not checkpoint.exists():
        if not checkpoint.exists():
            print("[!] Checkpoint not found. Training model from scratch...")
        train_model_pipeline(model, train_loader, val_loader, train_label, checkpoint)
    else:
        model = checkpoint.load(model)
        model.to(device)

    evaluate_model(
        model,
        val_loader,
        val_label,
        class_names=["Non-Seizure", "Seizure"],
        save_path=f"./report/{model_name}_confusion_matrix.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model before evaluation')
    parser.add_argument('--model', type=str, default='transformer', help='Model to use (e.g., transformer, cnn, rnn)')
    args = parser.parse_args()

    main(train=args.train, model_name=args.model)
