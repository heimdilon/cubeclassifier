"""Transfer learning model using MobileNetV3 for small datasets."""

import torch
import torch.nn as nn
from torchvision import models

if __package__:
    from .utils import logger
else:
    from utils import logger


class TransferLearningCubeClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        logger.info("Loading MobileNetV3 Small with ImageNet1K weights")
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            progress=False,
        )

        original_first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None,
        )

        with torch.no_grad():
            original_weights = original_first_conv.weight.data
            grayscale_weights = original_weights.mean(dim=1, keepdim=True)
            self.backbone.features[0][0].weight.data = grayscale_weights
            if original_first_conv.bias is not None:
                self.backbone.features[0][
                    0
                ].bias.data = original_first_conv.bias.data.clone()

        num_ftrs = self.backbone.classifier[0].in_features
        logger.info(f"Backbone feature dimension: {num_ftrs}")

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 64),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )
        self.backbone.classifier = self.classifier

        if freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in self.backbone.features.parameters():
                param.requires_grad = False

            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        else:
            logger.warning(
                "Backbone is unfrozen; overfitting risk is high on very small datasets."
            )

    def forward(self, x):
        return self.backbone(x)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model statistics")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"Trainable percent: {100 * trainable_params / total_params:.2f}%")

    return total_params, trainable_params


if __name__ == "__main__":
    logger.info("Creating transfer learning model")
    model = TransferLearningCubeClassifier(num_classes=2, freeze_backbone=True)
    count_parameters(model)

    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)

    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info("Model creation successful")
