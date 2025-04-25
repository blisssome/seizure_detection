import torch
from torch.utils.data import (Dataset,
                              DataLoader)

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        """
        eeg_data: np.array of shape (n_trials, n_channels, n_samples)
        labels: np.array of shape (n_trials,) or (n_trials, 1)
        """
        self.eeg = torch.tensor(eeg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx]
    
    def create_loader(self):
        return DataLoader(self, batch_size=128)
    
if __name__ == "__main__":
    from loader import read_data
    
    train_eeg, train_label, val_eeg, val_label = read_data()
    
    train_dataset = EEGDataset(train_eeg, train_label)
    val_dataset = EEGDataset(val_eeg, val_label)
    
    train_loader = train_dataset.create_loader()
    val_loader = val_dataset.create_loader()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print(f"Amount of train batches: {len(train_loader)}")
    print(f"Amount of validation batches: {len(val_loader)}")
    