import numpy as np

from scipy.signal import resample

from config import (TRAIN_DATA_PATH, 
                    VAL_DATA_PATH)

def read_data():
    """
    Loads EEG training and validation data from the specified NumPy archive files.

    Returns:
        train_eeg (np.ndarray): EEG signals for the training set, shape (n_trials, n_channels, n_samples).
        train_label (np.ndarray): Corresponding labels for the training EEG signals.
        val_eeg (np.ndarray): EEG signals for the validation set, shape (n_trials, n_channels, n_samples).
        val_label (np.ndarray): Corresponding labels for the validation EEG signals.
    """
    train_data = np.load(TRAIN_DATA_PATH)
    val_data = np.load(VAL_DATA_PATH)
    
    train_eeg, train_label = train_data["train_signals"], train_data["train_labels"]
    val_eeg, val_label = val_data["val_signals"], val_data["val_labels"]
    
    return train_eeg, train_label, val_eeg, val_label

def downsample_eeg(eeg_data, original_fs=256, target_fs=64):
    """
    eeg_data: numpy array of shape (n_trials, n_channels, n_samples)
    Returns: downsampled EEG (same shape but with fewer time points)
    """
    
    if original_fs == target_fs:
        return eeg_data

    n_trials, n_channels, n_samples = eeg_data.shape
    new_n_samples = int(n_samples * target_fs / original_fs)

    eeg_downsampled = np.zeros((n_trials, n_channels, new_n_samples), dtype=eeg_data.dtype)

    for trial in range(n_trials):
        for ch in range(n_channels):
            eeg_downsampled[trial, ch] = resample(eeg_data[trial, ch], new_n_samples)

    return eeg_downsampled

if __name__ == "__main__":
    train_eeg, train_label, val_eeg, val_label = read_data()

    print(f"Train eeg shape: {train_eeg.shape}")
    print(f"Validation eeg shape: {val_eeg.shape}")

    print(f"Train label shape: {train_label.shape}")
    print(f"Validation label shape: {val_label.shape}")
    
    
    

