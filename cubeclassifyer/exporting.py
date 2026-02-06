import os

import torch
import torch.nn as nn

if __package__:
    from . import config
    from .checkpoint_utils import safe_torch_load
    from .modeling import LightweightCubeClassifier
    from .utils import logger
else:
    import config
    from checkpoint_utils import safe_torch_load
    from modeling import LightweightCubeClassifier
    from utils import logger


def _instantiate_model(model_class):
    try:
        return model_class(num_classes=config.NUM_CLASSES)
    except TypeError:
        return model_class()


def _load_state_dict(model_path):
    loaded = safe_torch_load(model_path, map_location="cpu")
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    if isinstance(loaded, dict):
        return loaded
    raise ValueError("Model file does not contain a valid state_dict.")


def convert_model_for_rpi(
    model_path,
    output_path=config.TORCHSCRIPT_MODEL_PATH,
    model_class=LightweightCubeClassifier,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = _instantiate_model(model_class)
    model.load_state_dict(_load_state_dict(model_path))
    model.eval()

    example_input = torch.rand(1, 1, config.INPUT_HEIGHT, config.INPUT_WIDTH)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_path)

    logger.info(f"TorchScript model exported to '{output_path}'")
    return output_path


def export_onnx_for_rpi(
    model_path,
    output_path=config.ONNX_MODEL_PATH,
    model_class=LightweightCubeClassifier,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = _instantiate_model(model_class)
    model.load_state_dict(_load_state_dict(model_path))
    model.eval()

    example_input = torch.rand(1, 1, config.INPUT_HEIGHT, config.INPUT_WIDTH)
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=13,
    )

    logger.info(f"ONNX model exported to '{output_path}'")
    return output_path


def export_quantized_torchscript_for_rpi(
    model_path,
    output_path=config.QUANTIZED_TORCHSCRIPT_MODEL_PATH,
    model_class=LightweightCubeClassifier,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = _instantiate_model(model_class)
    model.load_state_dict(_load_state_dict(model_path))
    model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    example_input = torch.rand(1, 1, config.INPUT_HEIGHT, config.INPUT_WIDTH)
    traced_quantized_model = torch.jit.trace(quantized_model, example_input)
    traced_quantized_model.save(output_path)

    logger.info(f"Quantized TorchScript model exported to '{output_path}'")
    return output_path
