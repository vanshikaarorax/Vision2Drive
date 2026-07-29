"""
transforms.py

Image preprocessing and augmentation for the Vision2Drive dataset.

This module defines torchvision transformation pipelines for:
    - Training
    - Validation
    - Testing

Only RGB images are transformed.
LiDAR BEV maps and state vectors are processed separately.

Author: Vanshika
"""

from torchvision import transforms


def get_train_transform():
    """
    Returns the image transformation pipeline used during training.

    Steps:
        - Convert NumPy array to PIL Image
        - Resize image to 224x224
        - Apply light color augmentation
        - Convert image to PyTorch tensor
        - Normalize using ImageNet statistics
    """

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        # Light color augmentation.
        # Horizontal flipping is intentionally omitted because
        # autonomous driving labels (steering) are direction-sensitive.
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform():
    """
    Returns the image transformation pipeline used during validation.

    No data augmentation is applied.
    """

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transform():
    """
    Returns the image transformation pipeline used during testing.

    Identical to the validation pipeline.
    """

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])