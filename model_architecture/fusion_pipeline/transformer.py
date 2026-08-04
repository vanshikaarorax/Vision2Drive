"""
transformer.py

Transformer-based multimodal fusion for Vision2Drive.

This module fuses RGB and LiDAR feature maps using cross-attention.

Author: Vanshika
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings for transformer tokens.
    """

    def __init__(
        self,
        num_tokens: int = 49,
        embed_dim: int = 512,
    ):
        """
        Args:
            num_tokens:
                Number of spatial tokens.

            embed_dim:
                Feature dimension.
        """

        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional embeddings.

        Args:
            x:
                Token sequence.
                Shape: (B, N, D)

        Returns:
            Position-aware tokens.
        """

        return x + self.position_embedding
    
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention between two modalities.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim:
                Token embedding dimension.

            num_heads:
                Number of attention heads.

            dropout:
                Attention dropout.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Args:
            query:
                Query tokens.

            key:
                Key tokens.

            value:
                Value tokens.

        Returns:
            Cross-attended features.
        """

        output, _ = self.attention(
            query=query,
            key=key,
            value=value,
        )

        return output
class TransformerBlock(nn.Module):
    """
    Transformer block for multimodal feature fusion.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim:
                Token dimension.

            num_heads:
                Number of attention heads.

            mlp_ratio:
                Hidden dimension expansion ratio.

            dropout:
                Dropout probability.
        """

        super().__init__()

        self.cross_attention = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        lidar_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-modal transformer block.

        Args:
            rgb_tokens:
                RGB token sequence.

            lidar_tokens:
                LiDAR token sequence.

        Returns:
            Fused token sequence.
        """

        # Cross Attention
        attention = self.cross_attention(
            query=rgb_tokens,
            key=lidar_tokens,
            value=lidar_tokens,
        )

        rgb_tokens = self.norm1(
            rgb_tokens + attention
        )

        # Feed Forward
        mlp_output = self.mlp(rgb_tokens)

        output = self.norm2(
            rgb_tokens + mlp_output
        )

        return output
    
class TransformerFusion(nn.Module):
    """
    Transformer-based RGB-LiDAR fusion.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_tokens: int = 49,
    ):
        super().__init__()

        self.position = PositionalEncoding(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse RGB and LiDAR feature maps.

        Args:
            rgb_features:
                (B,512,7,7)

            lidar_features:
                (B,512,7,7)

        Returns:
            Fused feature map.
        """

        B, C, H, W = rgb_features.shape

        # Feature maps → tokens
        rgb_tokens = (
            rgb_features.flatten(2)
            .transpose(1, 2)
        )

        lidar_tokens = (
            lidar_features.flatten(2)
            .transpose(1, 2)
        )

        # Add positional embeddings
        rgb_tokens = self.position(rgb_tokens)
        lidar_tokens = self.position(lidar_tokens)

        # Transformer layers
        fused = rgb_tokens

        for block in self.blocks:
            fused = block(
                fused,
                lidar_tokens,
            )

        # Tokens → feature maps
        fused = (
            fused.transpose(1, 2)
            .reshape(B, C, H, W)
        )

        return fused
    

if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive Transformer Fusion Test")
    print("=" * 60)

    # Create transformer
    transformer = TransformerFusion(
        embed_dim=512,
        num_heads=8,
        num_layers=2,
    )

    # Dummy feature maps
    rgb = torch.randn(
        2,
        512,
        7,
        7,
    )

    lidar = torch.randn(
        2,
        512,
        7,
        7,
    )

    # Forward pass
    fused = transformer(
        rgb,
        lidar,
    )

    print(f"RGB Shape      : {rgb.shape}")
    print(f"LiDAR Shape    : {lidar.shape}")
    print(f"Output Shape   : {fused.shape}")

    # Parameter statistics
    total_params = sum(
        p.numel()
        for p in transformer.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in transformer.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Verify output dimensions
    expected_shape = (
        rgb.shape[0],
        512,
        7,
        7,
    )

    print(
        f"\nOutput Correct : {fused.shape == expected_shape}"
    )

    print("\nTransformer Fusion initialized successfully.")