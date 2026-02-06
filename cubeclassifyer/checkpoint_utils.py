import torch


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Secure checkpoint loading requires PyTorch >= 2.0 with weights_only support."
        ) from exc
