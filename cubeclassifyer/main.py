import torch
import os
from cube_classifier import (
    LightweightCubeClassifier,
    train_model,
    CubeDataset,
    get_transforms,
    convert_model_for_rpi,
)
from torch.utils.data import DataLoader
import config
import argparse
from utils import logger


def prepare_data():
    """Prepare data directories structure"""
    data_dirs = [
        os.path.join(config.TRAIN_DIR, "good"),
        os.path.join(config.TRAIN_DIR, "defective"),
        os.path.join(config.VAL_DIR, "good"),
        os.path.join(config.VAL_DIR, "defective"),
    ]

    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)

    logger.info("Data directory structure created:")
    logger.info(f"- {config.DATA_DIR}/")
    logger.info(f"  - train/")
    logger.info(f"    - good/")
    logger.info(f"    - defective/")
    logger.info(f"  - val/")
    logger.info(f"    - good/")
    logger.info(f"    - defective/")
    logger.info(
        "\nPlease place your 320x240 grayscale cube images in the appropriate folders."
    )

    print("Data directory structure created:")
    print("- cube_dataset/")
    print("  - train/")
    print("    - good/")
    print("    - defective/")
    print("  - val/")
    print("    - good/")
    print("    - defective/")
    print(
        "\nPlease place your 320x240 grayscale cube images in the appropriate folders."
    )


def train_cube_classifier(resume_from=None):
    """Train the cube classifier model

    Args:
        resume_from: Path to checkpoint file to resume training from
    """
    # Check if data directories exist
    if not os.path.exists(config.DATA_DIR):
        logger.info("Data directories not found. Creating structure...")
        prepare_data()

    # Create datasets
    train_dataset = CubeDataset(
        root_dir=config.TRAIN_DIR, transform=get_transforms(train=True)
    )

    val_dataset = CubeDataset(
        root_dir=config.VAL_DIR, transform=get_transforms(train=False)
    )

    # Check if we have data
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.warning("No data found in dataset directories.")
        logger.info(
            "Please place your 320x240 grayscale cube images in appropriate folders:"
        )
        logger.info(f"- {os.path.join(config.TRAIN_DIR, 'good')}/")
        logger.info(f"- {os.path.join(config.TRAIN_DIR, 'defective')}/")
        logger.info(f"- {os.path.join(config.VAL_DIR, 'good')}/")
        logger.info(f"- {os.path.join(config.VAL_DIR, 'defective')}/")
        return

    # Create data loaders with optimization flags
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_PINNED_MEMORY,
        persistent_workers=config.USE_PERSISTENT_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_PINNED_MEMORY,
        persistent_workers=config.USE_PERSISTENT_WORKERS,
    )

    if config.USE_PINNED_MEMORY and torch.cuda.is_available():
        logger.info("Memory pinning enabled for faster GPU transfer")
    if config.USE_PERSISTENT_WORKERS:
        logger.info("Persistent workers enabled for faster data loading")

    # Create model
    model = LightweightCubeClassifier(num_classes=config.NUM_CLASSES)

    logger.info(
        f"Training model with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )
    logger.info(
        f"Configuration: epochs={config.NUM_EPOCHS}, lr={config.LEARNING_RATE}, batch_size={config.BATCH_SIZE}"
    )
    logger.info("Starting training...")

    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE,
        max_grad_norm=config.MAX_GRAD_NORM,
        resume_from_checkpoint=resume_from,
    )

    logger.info(f"Training completed! Model saved as '{config.MODEL_PATH}'")

    # Convert model for Raspberry Pi deployment
    if os.path.exists(config.MODEL_PATH):
        logger.info("Converting model for Raspberry Pi deployment...")
        try:
            convert_model_for_rpi(config.MODEL_PATH, config.TORCHSCRIPT_MODEL_PATH)
            logger.info(
                f"Model converted successfully! Transfer '{config.TORCHSCRIPT_MODEL_PATH}' to your Raspberry Pi."
            )
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            logger.info(
                f"You can manually convert later using: python -c \"from cube_classifier import convert_model_for_rpi; convert_model_for_rpi('{config.MODEL_PATH}')\""
            )
    else:
        logger.warning(
            f"Model file '{config.MODEL_PATH}' not found. Skipping conversion."
        )


def main():
    parser = argparse.ArgumentParser(description="Cube Classifier Training Pipeline")
    parser.add_argument(
        "action",
        choices=["prepare", "train"],
        help="Action to perform: 'prepare' or 'train'",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")

    logger.info("")

    if args.action == "prepare":
        prepare_data()
    elif args.action == "train":
        train_cube_classifier(resume_from=args.resume)


if __name__ == "__main__":
    main()
