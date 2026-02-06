"""Backward-compatible facade for cube classifier modules."""

if __package__:
    from .dataset import AlbumentationsWrapper, CubeDataset
    from .exporting import (
        convert_model_for_rpi,
        export_onnx_for_rpi,
        export_quantized_torchscript_for_rpi,
    )
    from .modeling import LightweightCubeClassifier
    from .training import get_transforms, train_model
else:
    from dataset import AlbumentationsWrapper, CubeDataset
    from exporting import (
        convert_model_for_rpi,
        export_onnx_for_rpi,
        export_quantized_torchscript_for_rpi,
    )
    from modeling import LightweightCubeClassifier
    from training import get_transforms, train_model


__all__ = [
    "AlbumentationsWrapper",
    "CubeDataset",
    "LightweightCubeClassifier",
    "get_transforms",
    "train_model",
    "convert_model_for_rpi",
    "export_onnx_for_rpi",
    "export_quantized_torchscript_for_rpi",
]
