"""
lidar_encoder.py

CNN-based LiDAR encoder for Vision2Drive.

This module converts a Bird's-Eye View (BEV) occupancy map into
a high-level spatial feature map using a modified ResNet-18.

Input:
    BEV Tensor
    Shape: (B, 1, 200, 200)

Output:
    Feature Map
    Shape: (B, 512, 7, 7)

Author: Vanshika
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class LiDAREncoder(nn.Module):
    """
    CNN-based feature extractor for LiDAR BEV images.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the LiDAR encoder.

        Args:
            pretrained:
                Whether to load ImageNet pretrained weights.
        """

        super().__init__()

        self.backbone = self._build_backbone(pretrained)

    def _build_backbone(
        self,
        pretrained: bool,
    ) -> nn.Module:
        """
        Build the modified ResNet-18 backbone.

        The first convolution is adapted to accept
        a single-channel BEV image.

        Returns:
            ResNet-18 feature extractor.
        """

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        resnet = models.resnet18(weights=weights)

        # Replace first convolution
        original_conv = resnet.conv1

        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize new convolution
        if pretrained:
            resnet.conv1.weight.data = (
                original_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                )
            )

        backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        return backbone
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the LiDAR encoder.

        Args:
            x:
                BEV occupancy tensor.
                Shape: (B, 1, H, W)

        Returns:
            High-level spatial feature map.
            Shape: (B, 512, 7, 7)
        """

        # Validate input dimensions
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, 1, H, W), got {tuple(x.shape)}"
            )

        if x.size(1) != 1:
            raise ValueError(
                f"Expected single-channel BEV input, got {x.size(1)} channels."
            )

        # Extract LiDAR features
        features = self.backbone(x)

        return features
if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive LiDAR Encoder Test")
    print("=" * 60)

    # Create encoder
    encoder = LiDAREncoder(pretrained=True)

    # Dummy BEV batch
    dummy_bev = torch.randn(
        2,
        1,
        200,
        200,
    )

    # Forward pass
    features = encoder(dummy_bev)

    print(f"Input Shape : {dummy_bev.shape}")
    print(f"Output Shape: {features.shape}")

    # Parameter counts
    total_params = sum(
        p.numel()
        for p in encoder.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Freeze test
    encoder.freeze_backbone()

    frozen = all(
        not p.requires_grad
        for p in encoder.backbone.parameters()
    )

    print(f"\nBackbone Frozen      : {frozen}")

    # Unfreeze test
    encoder.unfreeze_backbone()

    unfrozen = all(
        p.requires_grad
        for p in encoder.backbone.parameters()
    )

    print(f"Backbone Unfrozen    : {unfrozen}")

    print("\nLiDAR Encoder initialized successfully.")