"""Configuration file for cube classifier training"""

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/training.log"

# Reproducibility
SEED = 42

# Optimizations
USE_MIXED_PRECISION = True  # Enable AMP for faster training (GPU only)
USE_PINNED_MEMORY = True  # Pin memory for faster GPU transfer
USE_PERSISTENT_WORKERS = True  # Keep workers alive for faster data loading

# Training hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
PATIENCE = 10  # Early stopping patience
WEIGHT_DECAY = 0.01

# Scheduler parameters
SCHEDULER_ETA_MIN = 1e-6
WARMUP_EPOCHS = 2

# Data loading
NUM_WORKERS = 2  # Set to 0 on Windows if experiencing issues

# Gradient clipping
MAX_GRAD_NORM = 1.0  # Disable by setting to None

# Data augmentation
TRAIN_ROTATION_DEGREES = 30

# Model
NUM_CLASSES = 2
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CLASSIFIER_DROPOUT = 0.4

# Paths
DATA_DIR = "cube_dataset"
TRAIN_DIR = "cube_dataset/train"
VAL_DIR = "cube_dataset/val"
MODEL_PATH = "best_cube_classifier.pth"
TORCHSCRIPT_MODEL_PATH = "cube_classifier_rpi.pt"
QUANTIZED_TORCHSCRIPT_MODEL_PATH = "cube_classifier_rpi_int8.pt"
ONNX_MODEL_PATH = "cube_classifier_rpi.onnx"

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
