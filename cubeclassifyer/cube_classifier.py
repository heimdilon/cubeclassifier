import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import config
from utils import logger
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler


# Custom dataset class for cube detection with grayscale images
class CubeDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(240, 320)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Load annotations
        self.annotations = []
        annotations_path = os.path.join(root_dir, "annotations.json")
        if os.path.exists(annotations_path):
            with open(annotations_path, "r") as f:
                self.annotations = json.load(f)
        else:
            # If no annotations file, assume all images are in good/defective folders
            self.annotations = self._create_annotations_from_folders()

    def _create_annotations_from_folders(self):
        annotations = []
        for class_name in ["good", "defective"]:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        annotations.append(
                            {
                                "image_path": os.path.join(class_name, filename),
                                "label": 0
                                if class_name == "good"
                                else 1,  # 0: good, 1: defective
                            }
                        )
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann["image_path"])
        try:
            # Load as grayscale
            image = Image.open(img_path).convert("L")

            if self.transform:
                image = self.transform(image)

            label = torch.tensor(ann["label"], dtype=torch.long)
            return image, label
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a blank image (all zeros) as fallback
            image = Image.new("L", (320, 240), 0)
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(ann["label"], dtype=torch.long)
            return image, label
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return a blank image (all zeros) as fallback
            image = Image.new("L", (320, 240), 0)
            if self.transform:
                image = self.transform(image)
            label = torch.tensor(ann["label"], dtype=torch.long)
            return image, label


# Lightweight CNN for grayscale images
class LightweightCubeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightCubeClassifier, self).__init__()

        # Custom lightweight CNN for grayscale images
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# Data augmentation transforms for grayscale images
def get_transforms(train=True):
    """Get data transforms for training or validation

    Args:
        train: If True, includes data augmentation

    Returns:
        Composed transforms
    """
    # Base transforms (used in both train and val)
    base_transforms = [
        transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    # Add augmentation for training
    if train:
        augmentations = [
            transforms.RandomRotation(degrees=config.TRAIN_ROTATION_DEGREES),
        ]
        transforms_list = augmentations + base_transforms
    else:
        transforms_list = base_transforms

    return transforms.Compose(transforms_list)


# Training function
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    learning_rate=0.001,
    patience=5,
    max_grad_norm=1.0,
    resume_from_checkpoint=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        logger.info(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.4f}")
    else:
        best_acc = 0.0
        epochs_without_improvement = 0
        if resume_from_checkpoint:
            logger.warning(
                f"Checkpoint not found: {resume_from_checkpoint}. Starting from scratch."
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Enable mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    if use_amp:
        logger.info("Mixed precision training enabled (AMP)")

    # Use cosine annealing scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    logger.info("Using Cosine Annealing learning rate scheduler")

    # Calculate class weights for handling imbalance
    class_counts = torch.zeros(2)
    for _, labels in train_loader:
        class_counts += torch.bincount(labels, minlength=2)

    # Inverse weighting: weight = total_samples / (num_classes * class_count)
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)

    # Use weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    logger.info(f"Class distribution: {class_counts.numpy()}")
    logger.info(f"Class weights: {class_weights.numpy()}")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Use automatic mixed precision
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # Scale loss and backward with gradient scaler
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient clipping to prevent exploding gradients
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Step optimizer with gradient scaler
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        logger.info(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save best model with timestamp
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            epochs_without_improvement = 0

            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.dirname(config.MODEL_PATH) or "."
            model_base = os.path.splitext(os.path.basename(config.MODEL_PATH))[0]
            model_ext = os.path.splitext(config.MODEL_PATH)[1]
            timestamped_path = os.path.join(
                model_dir, f"{model_base}_{timestamp}{model_ext}"
            )

            torch.save(model.state_dict(), config.MODEL_PATH)
            torch.save(model.state_dict(), timestamped_path)
            logger.info(f"New best model saved! Validation accuracy: {best_acc:.4f}")
            logger.info(f"Model saved as: {config.MODEL_PATH} and {timestamped_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            logger.info(f"Best validation accuracy: {best_acc:.4f}")
            break

        # Save checkpoint every N epochs
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
                    "best_acc": best_acc,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info(f"Training complete. Best val Acc: {best_acc:.4f}")
    return model


# Function to convert model for Raspberry Pi deployment
def convert_model_for_rpi(model_path):
    """Convert trained model to TorchScript for Raspberry Pi deployment

    Args:
        model_path: Path to the trained model (.pth file)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = LightweightCubeClassifier()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Convert to TorchScript for easier deployment
        example_input = torch.rand(1, 1, 240, 320)  # Grayscale input
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("cube_classifier_rpi.pt")

        print("Model converted for Raspberry Pi deployment!")
        print(f"Output file: cube_classifier_rpi.pt")
    except Exception as e:
        print(f"Error converting model: {e}")
        raise

    logger.info(
        f"Model converted for Raspberry Pi deployment! Saved to '{output_path}'"
    )
