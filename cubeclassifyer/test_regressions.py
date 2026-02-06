import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import models as tv_models

if __package__:
    from . import rpi_cube_detector as detector
    from . import transfer_learning_model as tlm
    from .cube_classifier import CubeDataset, LightweightCubeClassifier, get_transforms
    from .generate_synthetic import (
        add_combined_defects,
        add_crack,
        add_dent,
        add_scratch,
        add_stain,
    )
else:
    import rpi_cube_detector as detector
    import transfer_learning_model as tlm
    from cube_classifier import CubeDataset, LightweightCubeClassifier, get_transforms
    from generate_synthetic import (
        add_combined_defects,
        add_crack,
        add_dent,
        add_scratch,
        add_stain,
    )


TransferLearningCubeClassifier = tlm.TransferLearningCubeClassifier


class GenerateSyntheticRegressionTests(unittest.TestCase):
    def _make_image(self):
        return np.full((224, 224), 180, dtype=np.uint8)

    def test_add_scratch_does_not_raise_and_preserves_shape(self):
        image = self._make_image()
        output = add_scratch(image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, image.dtype)

    def test_add_crack_does_not_raise_and_preserves_shape(self):
        image = self._make_image()
        output = add_crack(image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, image.dtype)

    def test_add_stain_does_not_raise_and_preserves_shape(self):
        image = self._make_image()
        for _ in range(10):
            output = add_stain(image)
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, image.dtype)

    def test_add_dent_does_not_raise_and_preserves_shape(self):
        image = self._make_image()
        for _ in range(10):
            output = add_dent(image)
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, image.dtype)

    def test_add_combined_defects_does_not_raise(self):
        image = self._make_image()
        for _ in range(20):
            output = add_combined_defects(image)
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, image.dtype)


class DatasetRegressionTests(unittest.TestCase):
    def test_dataset_filters_corrupt_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            good_dir = os.path.join(tmp_dir, "good")
            defective_dir = os.path.join(tmp_dir, "defective")
            os.makedirs(good_dir, exist_ok=True)
            os.makedirs(defective_dir, exist_ok=True)

            valid_image = Image.fromarray(np.full((224, 224), 127, dtype=np.uint8))
            valid_path = os.path.join(good_dir, "good_001.jpg")
            valid_image.save(valid_path)

            corrupt_path = os.path.join(defective_dir, "bad_001.jpg")
            with open(corrupt_path, "wb") as handle:
                handle.write(b"not-an-image")

            dataset = CubeDataset(tmp_dir, transform=get_transforms(train=False))
            self.assertEqual(len(dataset), 1)

            tensor, label = dataset[0]
            self.assertEqual(tuple(tensor.shape), (1, 224, 224))
            self.assertEqual(int(label.item()), 0)

    def test_dataset_rejects_path_traversal_annotations(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotations_path = os.path.join(tmp_dir, "annotations.json")
            with open(annotations_path, "w", encoding="utf-8") as handle:
                handle.write('[{"image_path": "../outside.jpg", "label": 0}]')

            dataset = CubeDataset(tmp_dir, transform=get_transforms(train=False))
            self.assertEqual(len(dataset), 0)


class NormalizationConsistencyTests(unittest.TestCase):
    def test_rpi_normalization_matches_training(self):
        for p in [0, 64, 127, 128, 192, 255]:
            train_val = (p / 255.0 - 0.5) / 0.5
            frame = np.full((224, 224, 3), p, dtype=np.uint8)
            rpi_val = detector.preprocess_image(frame)[0, 0, 0, 0]
            self.assertAlmostEqual(float(train_val), float(rpi_val), places=5)


class LightweightModelRegressionTests(unittest.TestCase):
    def test_lightweight_model_output_shape(self):
        model = LightweightCubeClassifier(num_classes=2)
        output = model(torch.randn(2, 1, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 2))


class RpiDetectorRegressionTests(unittest.TestCase):
    def test_main_returns_non_zero_on_missing_model(self):
        with patch(
            "sys.argv",
            [
                "rpi_cube_detector.py",
                "--model",
                "this_file_does_not_exist.pt",
            ],
        ):
            exit_code = detector.main()
        self.assertEqual(exit_code, 1)


class TransferLearningRegressionTests(unittest.TestCase):
    def test_transfer_model_output_matches_num_classes(self):
        original_builder = tv_models.mobilenet_v3_small

        def _local_builder(*_args, **_kwargs):
            return original_builder(weights=None, progress=False)

        with patch.object(
            tlm.models,
            "mobilenet_v3_small",
            side_effect=_local_builder,
        ):
            model = TransferLearningCubeClassifier(num_classes=2, freeze_backbone=True)

        output = model(torch.randn(1, 1, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 2))

    def test_transfer_model_bias_param_is_bool(self):
        original_builder = tv_models.mobilenet_v3_small

        def _local_builder(*_args, **_kwargs):
            return original_builder(weights=None, progress=False)

        with patch.object(
            tlm.models,
            "mobilenet_v3_small",
            side_effect=_local_builder,
        ):
            model = TransferLearningCubeClassifier(num_classes=2, freeze_backbone=True)

        first_conv = model.backbone.features[0][0]
        self.assertEqual(first_conv.in_channels, 1)


if __name__ == "__main__":
    unittest.main()
