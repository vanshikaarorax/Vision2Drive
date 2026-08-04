"""
heads.py

Prediction heads for Vision2Drive.

Converts the fused driving representation into
vehicle control commands.

Outputs:
    - Steering
    - Throttle
    - Brake

Author: Vanshika
"""

import torch
import torch.nn as nn


class DrivingHead(nn.Module):
    """
    Predicts vehicle control commands.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
    ):
        """
        Args:
            input_dim:
                Driving representation dimension.

            hidden_dim:
                Hidden layer size.
        """

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.steering_head = nn.Linear(
            hidden_dim,
            1,
        )

        self.throttle_head = nn.Linear(
            hidden_dim,
            1,
        )

        self.brake_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict vehicle controls.

        Args:
            x:
                Driving representation.
                Shape: (B,512)

        Returns:
            Dictionary containing
            steering,
            throttle,
            brake.
        """

        features = self.shared(x)

        steering = torch.tanh(
            self.steering_head(features)
        )

        throttle = torch.sigmoid(
            self.throttle_head(features)
        )

        brake = torch.sigmoid(
            self.brake_head(features)
        )

        return {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
        }
    

if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive Driving Head Test")
    print("=" * 60)

    head = DrivingHead()

    batch_size = 2

    features = torch.randn(
        batch_size,
        512,
    )

    outputs = head(features)

    print(f"Input Shape : {features.shape}")

    print("\nOutput Shapes")

    print(f"Steering : {outputs['steering'].shape}")
    print(f"Throttle : {outputs['throttle'].shape}")
    print(f"Brake    : {outputs['brake'].shape}")

    print("\nSample Predictions")

    print(f"Steering : {outputs['steering'].flatten()}")

    print(f"Throttle : {outputs['throttle'].flatten()}")

    print(f"Brake    : {outputs['brake'].flatten()}")

    total_params = sum(
        p.numel()
        for p in head.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in head.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")

    print(f"Trainable Parameters : {trainable_params:,}")

    print("\nDriving Head initialized successfully.")