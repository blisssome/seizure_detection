import os
import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", best_model_name="best_model.pt"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(checkpoint_dir, best_model_name)

    def save(self, model, name=None):
        path = self.best_model_path if name is None else os.path.join(self.checkpoint_dir, name)
        torch.save(model.state_dict(), path)
        print(f"[✓] Model saved to: {path}")

    def load(self, model, name=None):
        path = self.best_model_path if name is None else os.path.join(self.checkpoint_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        model.load_state_dict(torch.load(path))
        print(f"[✓] Model loaded from: {path}")
        return model

    def exists(self, name=None):
        """
        Checks if the specified checkpoint file exists.
        If no name is provided, checks for the default best model.
        """
        path = self.best_model_path if name is None else os.path.join(self.checkpoint_dir, name)
        return os.path.exists(path)