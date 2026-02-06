import os
from datetime import datetime
from typing import Dict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast

if __package__:
    from . import config
    from .checkpoint_utils import safe_torch_load
    from .utils import logger
else:
    import config
    from checkpoint_utils import safe_torch_load
    from utils import logger


def _compute_binary_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    positive_label: int = 1,
) -> Dict[str, float]:
    targets = targets.view(-1)
    predictions = predictions.view(-1)

    tp = torch.sum((targets == positive_label) & (predictions == positive_label)).item()
    fp = torch.sum((targets != positive_label) & (predictions == positive_label)).item()
    fn = torch.sum((targets == positive_label) & (predictions != positive_label)).item()
    tn = torch.sum((targets != positive_label) & (predictions != positive_label)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def get_transforms(train=True):
    resize = transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH))
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    if not train:
        return transforms.Compose([resize, transforms.ToTensor(), normalize])

    return transforms.Compose(
        [
            resize,
            transforms.RandomRotation(degrees=config.TRAIN_ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ]
    )


def _build_scheduler(optimizer, num_epochs: int):
    warmup_epochs = min(config.WARMUP_EPOCHS, max(0, num_epochs - 1))
    if warmup_epochs <= 0:
        logger.info("Using Cosine Annealing learning rate scheduler")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs),
            eta_min=config.SCHEDULER_ETA_MIN,
        )

    logger.info(
        "Using warmup + cosine scheduler "
        f"(warmup_epochs={warmup_epochs}, eta_min={config.SCHEDULER_ETA_MIN})"
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs - warmup_epochs),
                eta_min=config.SCHEDULER_ETA_MIN,
            ),
        ],
        milestones=[warmup_epochs],
    )


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    learning_rate=0.001,
    patience=5,
    max_grad_norm=1.0,
    resume_from_checkpoint=None,
    weight_decay=0.01,
):
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        logger.error("Training or validation dataset is empty; aborting training.")
        raise ValueError("Training or validation dataset is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_f1 = 0.0
    epochs_without_improvement = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    use_amp = torch.cuda.is_available() and config.USE_MIXED_PRECISION
    scaler = GradScaler("cuda") if use_amp else None

    if use_amp:
        logger.info("Mixed precision training enabled (AMP)")

    scheduler = _build_scheduler(optimizer, num_epochs)

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = safe_torch_load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if use_amp and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_f1 = checkpoint.get("best_f1", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        logger.info(
            "Resumed from epoch %s | best val loss %.4f | best val acc %.4f | best f1 %.4f",
            start_epoch,
            best_val_loss,
            best_val_acc,
            best_f1,
        )
    elif resume_from_checkpoint:
        logger.warning(
            f"Checkpoint not found: {resume_from_checkpoint}. Starting from scratch."
        )

    num_classes = config.NUM_CLASSES
    annotations = getattr(train_loader.dataset, "annotations", None)
    if annotations is not None:
        labels = torch.tensor(
            [int(ann["label"]) for ann in annotations], dtype=torch.long
        )
        class_counts = torch.bincount(labels, minlength=num_classes).to(torch.float)
    else:
        class_counts = torch.zeros(num_classes)
        for _, labels in train_loader:
            class_counts += torch.bincount(labels, minlength=num_classes).to(
                torch.float
            )

    total_samples = class_counts.sum()
    class_weights = torch.ones_like(class_counts)
    nonzero_classes = class_counts > 0
    if not torch.all(nonzero_classes):
        logger.warning(
            "One or more classes have zero samples; falling back to uniform weights."
        )
    class_weights[nonzero_classes] = total_samples / (
        num_classes * class_counts[nonzero_classes]
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    logger.info(f"Class distribution: {class_counts.numpy()}")
    logger.info(f"Class weights: {class_weights.numpy()}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels).item()
                val_predictions.append(preds.cpu())
                val_targets.append(labels.cpu())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects / len(val_loader.dataset)

        if val_predictions:
            val_predictions_tensor = torch.cat(val_predictions)
            val_targets_tensor = torch.cat(val_targets)
            metrics = _compute_binary_metrics(
                val_targets_tensor, val_predictions_tensor
            )
        else:
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "tn": 0.0,
            }

        logger.info(
            "Val Loss: %.4f Acc: %.4f Precision: %.4f Recall: %.4f F1: %.4f",
            val_epoch_loss,
            val_epoch_acc,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        logger.info(
            "Val Confusion Matrix | TP: %.0f FP: %.0f FN: %.0f TN: %.0f",
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
            metrics["tn"],
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc
            best_f1 = metrics["f1"]
            epochs_without_improvement = 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.dirname(config.MODEL_PATH) or "."
            model_base = os.path.splitext(os.path.basename(config.MODEL_PATH))[0]
            model_ext = os.path.splitext(config.MODEL_PATH)[1]
            timestamped_path = os.path.join(
                model_dir, f"{model_base}_{timestamp}{model_ext}"
            )

            torch.save(model.state_dict(), config.MODEL_PATH)
            torch.save(model.state_dict(), timestamped_path)
            logger.info(
                "New best model saved! Val loss: %.4f | val acc: %.4f | val f1: %.4f",
                best_val_loss,
                best_val_acc,
                best_f1,
            )
            logger.info(
                "Model saved as: %s and %s", config.MODEL_PATH, timestamped_path
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                f"No val loss improvement for {epochs_without_improvement} epoch(s)"
            )

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            logger.info(
                "Best metrics | Val loss: %.4f | Val acc: %.4f | Val f1: %.4f",
                best_val_loss,
                best_val_acc,
                best_f1,
            )
            break

        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "best_f1": best_f1,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info(
        "Training complete. Best metrics | Val loss: %.4f | Val acc: %.4f | Val f1: %.4f",
        best_val_loss,
        best_val_acc,
        best_f1,
    )
    return model
