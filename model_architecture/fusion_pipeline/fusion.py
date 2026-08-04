"""
fusion.py

Feature fusion module for Vision2Drive.

This module combines transformer features with vehicle
state and navigation information to produce the final
driving representation.

Author: Vanshika
"""

import torch
import torch.nn as nn


class VehicleStateEncoder(nn.Module):
    """
    Encodes low-dimensional vehicle state.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim:
                Number of vehicle state values.

            hidden_dim:
                Hidden layer size.

            output_dim:
                Encoded feature dimension.
        """

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode vehicle state.

        Args:
            state:
                Shape: (B, input_dim)

        Returns:
            Shape: (B, output_dim)
        """

        return self.encoder(state)
class NavigationEncoder(nn.Module):
    """
    Encodes navigation commands.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim:
                Number of navigation commands.

            hidden_dim:
                Hidden layer size.

            output_dim:
                Encoded feature dimension.
        """

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode navigation command.

        Args:
            command:
                Shape: (B, input_dim)

        Returns:
            Shape: (B, output_dim)
        """

        return self.encoder(command)
    
class FeatureFusion(nn.Module):
    """
    Combines transformer, vehicle state,
    and navigation features.
    """

    def __init__(
        self,
        transformer_dim: int = 512,
        vehicle_dim: int = 128,
        navigation_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        fusion_dim = (
            transformer_dim +
            vehicle_dim +
            navigation_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, transformer_dim),
            nn.ReLU(inplace=True),
        )
    def forward(
        self,
        transformer_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        navigation_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse all features.

        Args:
            transformer_features:
                Shape (B,512)

            vehicle_features:
                Shape (B,128)

            navigation_features:
                Shape (B,128)

        Returns:
            Driving representation.
            Shape (B,512)
        """

        fused = torch.cat(
            (
                transformer_features,
                vehicle_features,
                navigation_features,
            ),
            dim=1,
        )

        fused = self.fusion(fused)

        return fused
    

if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive Feature Fusion Test")
    print("=" * 60)

    vehicle_encoder = VehicleStateEncoder()

    navigation_encoder = NavigationEncoder()

    fusion = FeatureFusion()

    batch_size = 2

    transformer = torch.randn(
        batch_size,
        512,
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

    vehicle_features = vehicle_encoder(vehicle)

    navigation_features = navigation_encoder(
        navigation
    )

    output = fusion(
        transformer,
        vehicle_features,
        navigation_features,
    )

    print(f"Transformer Features : {transformer.shape}")

    print(f"Vehicle Features     : {vehicle_features.shape}")

    print(f"Navigation Features  : {navigation_features.shape}")

    print(f"Driving Features     : {output.shape}")

    total = sum(
        p.numel()
        for p in (
            list(vehicle_encoder.parameters())
            + list(navigation_encoder.parameters())
            + list(fusion.parameters())
        )
    )

    print(f"\nTotal Parameters : {total:,}")

    print("\nFeature Fusion initialized successfully.")