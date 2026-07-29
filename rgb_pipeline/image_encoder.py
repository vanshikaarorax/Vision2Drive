"""
image_encoder.py

RGB image encoder for Vision2Drive.

This module converts an RGB image into a high-level feature map using a
pretrained ResNet-18 backbone. The extracted features are later fused
with LiDAR features inside the multimodal transformer.

Input:
    RGB Image Tensor
    Shape: (B, 3, 224, 224)

Output:
    Feature Map
    Shape: (B, 512, 7, 7)

Author: Vanshika
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ImageEncoder(nn.Module):
    """
    CNN-based RGB feature extractor.

    Uses a pretrained ResNet-18 model with the final average pooling and
    classification layers removed, returning spatial feature maps instead
    of classification logits.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the RGB encoder.

        Args:
            pretrained (bool):
                If True, load ImageNet pretrained weights.
        """
        super().__init__()

        self.backbone = self._build_backbone(pretrained)

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        """
        Build the ResNet-18 feature extractor.

        Args:
            pretrained (bool):
                Whether to load ImageNet pretrained weights.

        Returns:
            nn.Module:
                ResNet-18 backbone without the average pooling
                and fully connected classification layers.
        """

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        resnet = models.resnet18(weights=weights)

        # Keep only convolutional feature extraction layers
        backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        return backbone
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.

        Useful during the initial stage of transfer learning where
        only newly added layers are trained.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters.

        Enables end-to-end fine-tuning after the newly added
        network components have stabilized.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RGB encoder.

        Args:
            x (torch.Tensor):
                RGB image tensor.
                Shape: (B, 3, 224, 224)

        Returns:
            torch.Tensor:
                High-level image feature map.
                Shape: (B, 512, 7, 7)
        """

        # Validate input dimensions
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}"
            )

        if x.size(1) != 3:
            raise ValueError(
                f"Expected 3-channel RGB input, got {x.size(1)} channels."
            )

        # Extract visual features
        features = self.backbone(x)

        return features
    

if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive RGB Image Encoder Test")
    print("=" * 60)

    # Create encoder
    encoder = ImageEncoder(pretrained=True)

    # Create dummy RGB batch
    dummy_images = torch.randn(2, 3, 224, 224)

    # Forward pass
    features = encoder(dummy_images)

    print(f"Input Shape : {dummy_images.shape}")
    print(f"Output Shape: {features.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(
        p.numel() for p in encoder.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Verify freeze / unfreeze utilities
    encoder.freeze_backbone()

    frozen = all(
        not p.requires_grad
        for p in encoder.backbone.parameters()
    )

    print(f"\nBackbone Frozen      : {frozen}")

    encoder.unfreeze_backbone()

    unfrozen = all(
        p.requires_grad
        for p in encoder.backbone.parameters()
    )

    print(f"Backbone Unfrozen    : {unfrozen}")

    print("\nRGB Encoder initialized successfully.")