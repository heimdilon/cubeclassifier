import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__:
    from . import config
    from .dataset import CubeDataset
    from .exporting import (
        convert_model_for_rpi,
        export_onnx_for_rpi,
        export_quantized_torchscript_for_rpi,
    )
    from .modeling import LightweightCubeClassifier
    from .training import get_transforms, train_model
    from .utils import configure_logger, logger
else:
    import config
    from dataset import CubeDataset
    from exporting import (
        convert_model_for_rpi,
        export_onnx_for_rpi,
        export_quantized_torchscript_for_rpi,
    )
    from modeling import LightweightCubeClassifier
    from training import get_transforms, train_model
    from utils import configure_logger, logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(_worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data():
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
    logger.info("  - train/")
    logger.info("    - good/")
    logger.info("    - defective/")
    logger.info("  - val/")
    logger.info("    - good/")
    logger.info("    - defective/")
    logger.info(
        "Please place your 224x224 grayscale cube images in the appropriate folders."
    )


def _build_dataloaders(train_dataset, val_dataset, seed: int):
    num_workers = config.NUM_WORKERS
    if os.name == "nt":
        num_workers = 0
        logger.info("Windows detected; using num_workers=0 for DataLoader.")

    pin_memory = config.USE_PINNED_MEMORY and torch.cuda.is_available()
    persistent_workers = config.USE_PERSISTENT_WORKERS and num_workers > 0
    worker_init_fn = _seed_worker if num_workers > 0 else None

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=val_generator,
    )

    logger.info(
        "DataLoader settings: num_workers=%s, pin_memory=%s, persistent_workers=%s",
        num_workers,
        pin_memory,
        persistent_workers,
    )
    return train_loader, val_loader


def export_deployment_models(
    model_path: str,
    export_onnx: bool,
    export_quantized: bool,
):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return

    convert_model_for_rpi(model_path, output_path=config.TORCHSCRIPT_MODEL_PATH)

    if export_quantized:
        try:
            export_quantized_torchscript_for_rpi(
                model_path,
                output_path=config.QUANTIZED_TORCHSCRIPT_MODEL_PATH,
            )
        except Exception as exc:
            logger.warning(f"Quantized TorchScript export skipped: {exc}")

    if export_onnx:
        try:
            export_onnx_for_rpi(model_path, output_path=config.ONNX_MODEL_PATH)
        except Exception as exc:
            logger.warning(f"ONNX export skipped: {exc}")


def train_cube_classifier(
    resume_from=None,
    seed: int = config.SEED,
    export_onnx: bool = True,
    export_quantized: bool = True,
):
    set_seed(seed)

    if not os.path.exists(config.DATA_DIR):
        logger.info("Data directories not found. Creating structure...")
        prepare_data()

    train_dataset = CubeDataset(
        root_dir=config.TRAIN_DIR,
        transform=get_transforms(train=True),
    )
    val_dataset = CubeDataset(
        root_dir=config.VAL_DIR,
        transform=get_transforms(train=False),
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.warning("No data found in dataset directories.")
        logger.info(
            "Please place your 224x224 grayscale cube images in appropriate folders:"
        )
        logger.info(f"- {os.path.join(config.TRAIN_DIR, 'good')}/")
        logger.info(f"- {os.path.join(config.TRAIN_DIR, 'defective')}/")
        logger.info(f"- {os.path.join(config.VAL_DIR, 'good')}/")
        logger.info(f"- {os.path.join(config.VAL_DIR, 'defective')}/")
        return

    train_loader, val_loader = _build_dataloaders(train_dataset, val_dataset, seed=seed)

    model = LightweightCubeClassifier(num_classes=config.NUM_CLASSES)

    logger.info(
        "Training model with %s training samples and %s validation samples",
        len(train_dataset),
        len(val_dataset),
    )
    logger.info(
        "Configuration: epochs=%s, lr=%s, batch_size=%s, weight_decay=%s, seed=%s",
        config.NUM_EPOCHS,
        config.LEARNING_RATE,
        config.BATCH_SIZE,
        config.WEIGHT_DECAY,
        seed,
    )

    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        patience=config.PATIENCE,
        max_grad_norm=config.MAX_GRAD_NORM,
        resume_from_checkpoint=resume_from,
        weight_decay=config.WEIGHT_DECAY,
    )

    logger.info(f"Training completed. Model saved as '{config.MODEL_PATH}'")
    export_deployment_models(
        config.MODEL_PATH,
        export_onnx=export_onnx,
        export_quantized=export_quantized,
    )


def main():
    parser = argparse.ArgumentParser(description="Cube Classifier Training Pipeline")
    parser.add_argument(
        "action",
        choices=["prepare", "train", "export"],
        help="Action to perform: 'prepare', 'train', or 'export'",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help=f"Random seed (default: {config.SEED})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.MODEL_PATH,
        help=f"Model path to export (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Skip ONNX export",
    )
    parser.add_argument(
        "--no-quantized",
        action="store_true",
        help="Skip quantized TorchScript export",
    )

    args = parser.parse_args()

    configure_logger(log_file=config.LOG_FILE, level=config.LOG_LEVEL)

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")

    if args.action == "prepare":
        prepare_data()
    elif args.action == "train":
        train_cube_classifier(
            resume_from=args.resume,
            seed=args.seed,
            export_onnx=not args.no_onnx,
            export_quantized=not args.no_quantized,
        )
    elif args.action == "export":
        export_deployment_models(
            args.model_path,
            export_onnx=not args.no_onnx,
            export_quantized=not args.no_quantized,
        )


if __name__ == "__main__":
    main()
