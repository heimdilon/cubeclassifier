import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

if __package__:
    from .utils import logger
else:
    from utils import logger


class AlbumentationsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        transformed = self.transform(image=np.array(image, dtype=np.uint8))
        return transformed["image"]


class CubeDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.root_dir = root_dir
        self.root_dir_abs = os.path.realpath(root_dir)
        self.transform = transform
        self.target_size = target_size

        self.annotations = self._load_annotations()
        self.annotations = self._filter_valid_annotations(self.annotations)

        if not self.annotations:
            logger.warning(f"No valid samples found in '{self.root_dir}'.")

    def _load_annotations(self) -> List[Dict[str, int]]:
        annotations_path = os.path.join(self.root_dir, "annotations.json")
        if not os.path.exists(annotations_path):
            return self._create_annotations_from_folders()

        try:
            with open(annotations_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list):
                raise ValueError("annotations.json must contain a list of records.")
            return loaded
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                f"Failed to load annotations from '{annotations_path}' ({exc}); "
                "falling back to folder scan."
            )
            return self._create_annotations_from_folders()

    def _create_annotations_from_folders(self) -> List[Dict[str, int]]:
        annotations: List[Dict[str, int]] = []
        if not os.path.exists(self.root_dir):
            return annotations

        class_dirs = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            class_key = class_name.lower()
            if class_key == "good":
                class_dirs.append((class_name, 0))
            elif class_key.startswith("defective"):
                class_dirs.append((class_name, 1))

        for class_name, label in sorted(class_dirs):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    annotations.append(
                        {
                            "image_path": os.path.join(class_name, filename),
                            "label": int(label),
                        }
                    )

        return annotations

    def _resolve_image_path(self, image_rel_path: str) -> Optional[str]:
        normalized_rel_path = os.path.normpath(image_rel_path)
        image_abs_path = os.path.realpath(
            os.path.join(self.root_dir_abs, normalized_rel_path)
        )

        try:
            is_within_root = (
                os.path.commonpath([self.root_dir_abs, image_abs_path])
                == self.root_dir_abs
            )
        except ValueError:
            is_within_root = False

        if not is_within_root:
            return None
        return image_abs_path

    def _filter_valid_annotations(
        self, annotations: List[Dict[str, int]]
    ) -> List[Dict[str, int]]:
        valid_annotations = []
        for ann in annotations:
            image_rel_path = ann.get("image_path")
            label = ann.get("label")

            if not isinstance(image_rel_path, str):
                logger.warning(
                    f"Skipping malformed annotation without image path: {ann}"
                )
                continue

            if label not in (0, 1):
                logger.warning(
                    f"Skipping malformed annotation with invalid label: {ann}"
                )
                continue

            image_abs_path = self._resolve_image_path(image_rel_path)
            if image_abs_path is None:
                logger.warning(
                    "Skipping annotation with path outside dataset root: %s",
                    image_rel_path,
                )
                continue

            if not os.path.exists(image_abs_path):
                logger.warning(f"Skipping missing image file: {image_abs_path}")
                continue

            try:
                with Image.open(image_abs_path) as image_handle:
                    image_handle.verify()
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning(f"Skipping unreadable image '{image_abs_path}': {exc}")
                continue

            valid_annotations.append(
                {"image_path": image_rel_path, "label": int(label)}
            )

        return valid_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = self._resolve_image_path(ann["image_path"])
        if img_path is None:
            raise LookupError(
                f"Resolved image path is outside dataset root: {ann['image_path']}"
            )

        try:
            with Image.open(img_path) as image_handle:
                image = image_handle.convert("L")
        except (UnidentifiedImageError, OSError) as exc:
            raise LookupError(f"Failed to load image '{img_path}': {exc}") from exc

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(ann["label"], dtype=torch.long)
        return image, label
