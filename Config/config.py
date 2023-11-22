# config.py

import torch
# Training configuration
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
INPUT_SIZE = 3 * 48 * 48  # 3 channels, 48x48 image size
HIDDEN_SIZE = 50
OUTPUT_SIZE = 4  # Number of output classes

# Path configuration
TRAIN_DATA_PATH = './dataset/Train'
MODEL_SAVE_PATH = './model/best_model.pth'
MODEL_SAVE_PATH_EARLY_STOP = './model/best_model_stop.pth'
STOP_MODEL_SAVE_PATH = './model/best_model_stop.pth'
TEST_DATA_PATH = './dataset/Test'
CLASS_NAMES = ['Angry', 'Bored', 'Engaged', 'Neutral']

