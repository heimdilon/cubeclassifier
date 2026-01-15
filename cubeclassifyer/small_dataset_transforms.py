"""
Aggressive augmentation transforms for very small datasets
Designed for 50-1000 image datasets
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np


def get_extreme_transforms(train=True):
    """
    EXTREME augmentation for very small datasets (<100 images)

    For 225 total images (50 training), applies 6-8 transforms per image
    This creates ~1,400 effective training samples

    Args:
        train: If True, includes aggressive augmentation

    Returns:
        Composed Albumentations transform
    """
    # Base transforms (always applied)
    base_transforms = [
        A.Resize(240, 320),
        A.Normalize(mean=[0.5], std=[0.5]),
    ]

    if not train:
        return A.Compose(base_transforms + [ToTensorV2()])

    # AGGRESSIVE augmentation for training
    augmentations = [
        # ALWAYS apply these (high probability)
        # 1. Geometric transforms (create more variety)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,  # ±15% position shift
            scale_limit=0.25,  # ±25% size scale
            rotate_limit=45,  # ±45° rotation (was only 10°!)
            p=0.8,  # 80% of images get this
        ),
        # 2. Brightness/contrast (simulate lighting changes)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # ±30% brightness
            contrast_limit=0.3,  # ±30% contrast
            p=0.7,
        ),
        # 3. Noise (simulate camera sensor noise)
        A.OneOf(
            [
                A.GaussNoise(var_limit=(10, 50), p=1.0),  # Gaussian noise
                A.ISONoise(p=1.0),  # Camera ISO noise
            ],
            p=0.4,
        ),
        # SOMETIMES apply these (medium probability)
        # 4. Blur (simulate focus issues)
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        # 5. Cutout (forces model to use partial information)
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            fill_value=0,
            p=0.4,
        ),
        # 6. Grid dropout (prevents over-reliance on certain features)
        A.GridDropout(
            ratio=0.2,
            p=0.3,
        ),
        # 7. Elastic distortion (simulates slight shape changes)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            sigma_affine=50,
            p=0.3,
        ),
        # 8. Perspective (simulates different camera angles)
        A.Perspective(
            scale=(0.05, 0.1),
            p=0.3,
        ),
        # RARELY apply these (low probability)
        # 9. Color/Hue shifts
        A.OneOf(
            [
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                ),
                A.ToGray(p=1.0),
            ],
            p=0.2,
        ),
    ]

    transforms_list = augmentations + base_transforms + [ToTensorV2()]

    print(f"Created {len(augmentations)} augmentation stages")
    print("Expected effective samples: ~1,400 (from 50 real images)")

    return A.Compose(transforms_list)


def get_moderate_transforms(train=True):
    """
    MODERATE augmentation for medium datasets (500-1000 images)

    For 300-500 images, applies 3-4 transforms per image

    Args:
        train: If True, includes augmentation

    Returns:
        Composed Albumentations transform
    """
    base_transforms = [
        A.Resize(240, 320),
        A.Normalize(mean=[0.5], std=[0.5]),
    ]

    if not train:
        return A.Compose(base_transforms + [ToTensorV2()])

    augmentations = [
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=20,
            p=0.6,
        ),
        # Brightness/contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        # Noise
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        # Blur
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.2,
        ),
        # Cutout
        A.CoarseDropout(
            max_holes=6,
            max_height=12,
            max_width=12,
            min_holes=1,
            fill_value=0,
            p=0.3,
        ),
    ]

    transforms_list = augmentations + base_transforms + [ToTensorV2()]

    print(f"Created {len(augmentations)} augmentation stages (moderate)")
    print("Expected effective samples: ~200 (from 50 real images)")

    return A.Compose(transforms_list)


def test_augmentation():
    """Visualize augmentation pipeline"""
    import cv2
    from PIL import Image

    # Create a dummy grayscale image
    dummy = np.ones((240, 320), dtype=np.uint8) * 200

    # Apply transformation
    transform = get_extreme_transforms(train=True)

    # Convert PIL Image to Albumentations format
    from albumentations.core.bbox_utils import convert_bboxes_to_albumentations

    # Apply transformation multiple times
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Aggressive Augmentation Pipeline", fontsize=16)

    for i in range(10):
        augmented = transform(image=dummy)
        # Convert tensor back to numpy
        img_np = augmented["image"].numpy()
        img_np = img_np.squeeze()

        # Denormalize: (x * 0.5) + 0.5 = x * 0.5 + 0.5
        img_np = (img_np * 0.5 + 0.5) * 255
        img_np = img_np.astype(np.uint8)

        ax = axes[i // 5, i % 5]
        ax.imshow(img_np, cmap="gray")
        ax.set_title(f"Variant {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("augmentation_examples.png", dpi=100, bbox_inches="tight")
    print("Saved augmentation examples to 'augmentation_examples.png'")

    return transform


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_augmentation()
    else:
        print("Available transforms:")
        print("  get_extreme_transforms()  - For <100 images (YOUR CASE)")
        print("  get_moderate_transforms() - For 500-1000 images")
        print("\nUse: python small_dataset_transforms.py --test")
