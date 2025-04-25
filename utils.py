"""
Utility functions for analyzing and visualizing classification model performance.

This module provides several helpful tools for working with classification tasks:
- `get_class_weights`: Computes balanced class weights for imbalanced datasets.
- `print_classification_report`: Nicely formats and prints precision, recall, F1-score, and support.
- `plot_confusion_matrix`: Creates a normalized confusion matrix with per-class counts and percentages.
- `plot_history`: Visualizes training and validation loss/accuracy over epochs.

Each function is designed for integration into training pipelines and supports enhanced visualization for clearer model diagnostics.
"""


def get_device():
    """
    Returns the appropriate device ('cuda' if available, else 'cpu').

    Useful for ensuring the code runs on GPU when available.
    """
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_pipeline(model, 
                         train_loader, 
                         val_loader, 
                         labels, 
                         checkpoint):
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"
    class_weights = get_class_weights(labels).to(device)
    print("Class Weights:", class_weights)

    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=50,
        lr=1e-3,
        checkpoint_path=checkpoint.best_model_path
    )

    plot_history(history)

def evaluate_model(model,
                   val_loader, 
                   val_label, 
                   class_names, 
                   save_path):
    import torch
    import os 
    
    os.makedirs("report", exist_ok=True)
    
    y_true = val_label.copy()
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in val_loader:
            outputs = model(batch_x.to("cuda"))
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())

    print_classification_report(y_true, y_pred, class_names=class_names)
    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)

def get_class_weights(labels):
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    import torch
    
    classes = np.unique(labels)

    # Compute class weights
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )

    return torch.tensor(weights, dtype=torch.float32)

def print_classification_report(y_true, y_pred, class_names=None, round_digits=2):
    """
    Prints a beautifully formatted classification report to the terminal.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: Optional list of class names
    - round_digits: Number of decimal places
    """
    import pandas as pd
    from sklearn.metrics import classification_report
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    df = df[["precision", "recall", "f1-score", "support"]]
    df = df.round(round_digits)

    header = "Classification Report"
    print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)

    for index, row in df.iterrows():
        label = f"{index:<15}"
        precision = f"{row['precision']:>10.2f}"
        recall = f"{row['recall']:>10.2f}"
        f1 = f"{row['f1-score']:>10.2f}"
        support = f"{int(row['support']):>10}"
        print(f"{label}{precision}{recall}{f1}{support}")
    
    print("=" * 60)

def plot_confusion_matrix(y_true, 
                          y_pred, 
                          class_names=None, 
                          figsize=(8, 6), 
                          font_scale=1.4, 
                          save_path=None):
    """
    Plots a confusion matrix.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: list of class names
    - figsize: Figure size
    - font_scale: Controls font size
    - save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    raw_cm = confusion_matrix(y_true, y_pred)
    cm = raw_cm.astype("float") / raw_cm.sum(axis=1, keepdims=True)
    n_classes = raw_cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    sns.set(style="white")
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap: faded colors for off-diagonal, highlight diagonal
    cmap = sns.light_palette("#2a9d8f", as_cmap=True)

    sns.heatmap(cm, annot=False, cmap=cmap, fmt=".2f", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.6, linecolor='white', square=True)

    for i in range(n_classes):
        for j in range(n_classes):
            count = raw_cm[i, j]
            pct = cm[i, j]
            is_diagonal = i == j
            text = f"{count}\n({pct:.1%})"
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    fontsize=13,
                    fontweight='bold' if is_diagonal else 'normal',
                    color='white' if is_diagonal else 'black',
                    bbox=dict(facecolor='#264653', alpha=0.7, boxstyle="round,pad=0.3") if is_diagonal else None)

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[S] Saved to: {save_path}")

    plt.show()

def plot_history(history,
                 save_path=None):
    """
    Parameters:
    -   history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    colors = {
        "train": "#4C72B0",  # muted blue
        "val": "#DD8452",    # muted orange
    }

    # Plot Loss
    axs[0].plot(history["train_loss"], label="Train", color=colors["train"], linewidth=2.5)
    axs[0].plot(history["val_loss"], label="Validation", color=colors["val"], linewidth=2.5)
    axs[0].set_title("Loss Over Epochs", fontsize=16)
    axs[0].set_xlabel("Epoch", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    axs[1].plot(history["train_acc"], label="Train", color=colors["train"], linewidth=2.5)
    axs[1].plot(history["val_acc"], label="Validation", color=colors["val"], linewidth=2.5)
    axs[1].set_title("Accuracy Over Epochs", fontsize=16)
    axs[1].set_xlabel("Epoch", fontsize=14)
    axs[1].set_ylabel("Accuracy (%)", fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # Cute finishing touches
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Training History", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[S] Saved to: {save_path}")