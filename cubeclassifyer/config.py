"""Configuration file for cube classifier training"""

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/training.log"

# Optimizations
USE_MIXED_PRECISION = True  # Enable AMP for faster training (GPU only)
USE_PINNED_MEMORY = True  # Pin memory for faster GPU transfer
USE_PERSISTENT_WORKERS = True  # Keep workers alive for faster data loading

# Training hyperparameters
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 16
PATIENCE = 5  # Early stopping patience

# Scheduler parameters
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1

# Data loading
NUM_WORKERS = 2  # Set to 0 on Windows if experiencing issues

# Gradient clipping
MAX_GRAD_NORM = 1.0  # Disable by setting to None

# Data augmentation
TRAIN_ROTATION_DEGREES = 10

# Model
NUM_CLASSES = 2
INPUT_HEIGHT = 240
INPUT_WIDTH = 320

# Paths
DATA_DIR = "cube_dataset"
TRAIN_DIR = "cube_dataset/train"
VAL_DIR = "cube_dataset/val"
MODEL_PATH = "best_cube_classifier.pth"
TORCHSCRIPT_MODEL_PATH = "cube_classifier_rpi.pt"

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
