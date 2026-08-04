"""
dataloader.py

Creates PyTorch DataLoaders for the Vision2Drive dataset.

Provides train, validation, and test DataLoaders with
appropriate preprocessing, batching, and shuffling.

Author: Vanshika
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from config import OUTPUT_DIR
from dataset import Vision2DriveDataset
from transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)


def create_dataloader(
    dataset: Vision2DriveDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    Args:
        dataset: Vision2Drive dataset.
        batch_size: Number of samples per batch.
        shuffle: Shuffle dataset every epoch.
        num_workers: Number of worker processes.

    Returns:
        Configured DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_dataloaders(
    dataset_root: Path = OUTPUT_DIR,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Args:
        dataset_root: Vision2Drive dataset directory.
        batch_size: Batch size.
        train_ratio: Fraction of training samples.
        val_ratio: Fraction of validation samples.
        test_ratio: Fraction of testing samples.
        num_workers: Number of DataLoader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    # -----------------------------------------------------
    # Create full dataset
    # -----------------------------------------------------

    full_dataset = Vision2DriveDataset(
        dataset_root=dataset_root,
        transform=get_train_transform(),
    )

    total_samples = len(full_dataset)

    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # -----------------------------------------------------
    # Assign transforms
    # -----------------------------------------------------

    train_dataset.dataset.transform = get_train_transform()
    val_dataset.dataset.transform = get_val_transform()
    test_dataset.dataset.transform = get_test_transform()

    # -----------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )