"""Albumentations pipelines for small grayscale cube datasets."""

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

if __package__:
    from . import config
else:
    import config


def _base_transforms():
    return [
        A.Resize(config.INPUT_HEIGHT, config.INPUT_WIDTH),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
    ]


def get_extreme_transforms(train=True):
    if not train:
        return A.Compose(_base_transforms() + [ToTensorV2()])

    augmentations = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.25,
            rotate_limit=45,
            p=0.8,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7,
        ),
        A.OneOf(
            [
                A.GaussNoise(std_range=(10 / 255, 50 / 255), p=1.0),
                A.ISONoise(p=1.0),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=0,
            p=0.4,
        ),
        A.GridDropout(ratio=0.2, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        A.RandomGamma(gamma_limit=(70, 130), p=0.2),
    ]

    return A.Compose(augmentations + _base_transforms() + [ToTensorV2()])


def get_moderate_transforms(train=True):
    if not train:
        return A.Compose(_base_transforms() + [ToTensorV2()])

    augmentations = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(std_range=(5 / 255, 25 / 255), p=0.3),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.2,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(6, 12),
            hole_width_range=(6, 12),
            fill=0,
            p=0.3,
        ),
    ]

    return A.Compose(augmentations + _base_transforms() + [ToTensorV2()])


def test_augmentation():
    import matplotlib.pyplot as plt

    dummy = np.ones((config.INPUT_HEIGHT, config.INPUT_WIDTH), dtype=np.uint8) * 200
    transform = get_extreme_transforms(train=True)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Aggressive Augmentation Pipeline", fontsize=16)

    for i in range(10):
        augmented = transform(image=dummy)
        image_tensor = augmented["image"]
        image_np = image_tensor.numpy().squeeze()
        image_np = (image_np * 0.5 + 0.5) * 255
        image_np = image_np.astype(np.uint8)

        ax = axes[i // 5, i % 5]
        ax.imshow(image_np, cmap="gray")
        ax.set_title(f"Variant {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=100, bbox_inches="tight")
    return transform


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_augmentation()
    else:
        print("Available transforms:")
        print("  get_extreme_transforms()  - For <100 images")
        print("  get_moderate_transforms() - For 500-1000 images")
        print("\nUse: python small_dataset_transforms.py --test")
