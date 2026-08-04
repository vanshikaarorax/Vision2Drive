"""
model.py

Vision2Drive

Complete end-to-end autonomous driving model.

Pipeline

RGB
↓

Image Encoder

↓

Transformer

↓

Feature Fusion

↓

Driving Head

Author: Vanshika
"""

import torch
import torch.nn as nn

from image_encoder import ImageEncoder
from lidar_encoder import LiDAREncoder
from transformer import TransformerFusion
from fusion import (
    VehicleStateEncoder,
    NavigationEncoder,
    FeatureFusion,
)
from heads import DrivingHead


class Vision2Drive(nn.Module):
    """
    Complete autonomous driving model.
    """

    def __init__(self):
        """
        Initialize all Vision2Drive modules.
        """

        super().__init__()

        # RGB Encoder
        self.image_encoder = ImageEncoder()

        # LiDAR Encoder
        self.lidar_encoder = LiDAREncoder()

        # Multimodal Transformer
        self.transformer = TransformerFusion()

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # Vehicle State Encoder
        self.vehicle_encoder = VehicleStateEncoder()

        # Navigation Encoder
        self.navigation_encoder = NavigationEncoder()

        # Feature Fusion
        self.fusion = FeatureFusion()

        # Driving Head
        self.head = DrivingHead()

    def _pool_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert feature maps into feature vectors.

        Args:
            features:
                Shape (B,C,H,W)

        Returns:
            Shape (B,C)
        """

        features = self.pool(features)

        features = torch.flatten(
            features,
            start_dim=1,
        )

        return features
    def forward(
        self,
        rgb: torch.Tensor,
        lidar_bev: torch.Tensor,
        vehicle_state: torch.Tensor,
        navigation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of Vision2Drive.

        Args:
            rgb:
                RGB image.
                Shape (B,3,224,224)

            lidar_bev:
                BEV occupancy map.
                Shape (B,1,200,200)

            vehicle_state:
                Vehicle state vector.
                Shape (B,2)

            navigation:
                Navigation command.
                Shape (B,4)

        Returns:
            Steering, throttle and brake predictions.
        """

        # --------------------------------------------------
        # RGB Encoder
        # --------------------------------------------------

        rgb_features = self.image_encoder(rgb)

        # --------------------------------------------------
        # LiDAR Encoder
        # --------------------------------------------------

        lidar_features = self.lidar_encoder(
            lidar_bev
        )

        # --------------------------------------------------
        # Transformer Fusion
        # --------------------------------------------------

        fused_features = self.transformer(
            rgb_features,
            lidar_features,
        )

        # --------------------------------------------------
        # Global Average Pooling
        # --------------------------------------------------

        scene_features = self._pool_features(
            fused_features
        )

        # --------------------------------------------------
        # Encode Vehicle State
        # --------------------------------------------------

        vehicle_features = self.vehicle_encoder(
            vehicle_state
        )

        # --------------------------------------------------
        # Encode Navigation
        # --------------------------------------------------

        navigation_features = self.navigation_encoder(
            navigation
        )

        # --------------------------------------------------
        # Feature Fusion
        # --------------------------------------------------

        driving_features = self.fusion(
            scene_features,
            vehicle_features,
            navigation_features,
        )

        # --------------------------------------------------
        # Driving Head
        # --------------------------------------------------

        outputs = self.head(
            driving_features
        )

        return outputs
    

if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive Complete Model Test")
    print("=" * 60)

    model = Vision2Drive()

    batch_size = 2

    rgb = torch.randn(
        batch_size,
        3,
        224,
        224,
    )

    lidar = torch.randn(
        batch_size,
        1,
        200,
        200,
    )

    vehicle = torch.randn(
        batch_size,
        2,
    )

    navigation = torch.tensor(
        [
            [1,0,0,0],
            [0,1,0,0],
        ],
        dtype=torch.float32,
    )

    outputs = model(
        rgb,
        lidar,
        vehicle,
        navigation,
    )

    print(f"RGB Shape        : {rgb.shape}")
    print(f"LiDAR Shape      : {lidar.shape}")
    print(f"Vehicle Shape    : {vehicle.shape}")
    print(f"Navigation Shape : {navigation.shape}")

    print("\nOutputs")

    print(f"Steering : {outputs['steering'].shape}")
    print(f"Throttle : {outputs['throttle'].shape}")
    print(f"Brake    : {outputs['brake'].shape}")

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")

    print("\nVision2Drive initialized successfully.")