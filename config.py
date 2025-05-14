import os 

DATA_PATH = r'data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'eeg-seizure_train.npz')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'eeg-seizure_test.npz')
VAL_DATA_PATH = os.path.join(DATA_PATH, 'eeg-seizure_val.npz')
