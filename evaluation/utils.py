# ============================================================
# Utility Functions
# ============================================================

import os
import json
import random
import numpy as np
import torch


# ============================================================
# Random Seed
# ============================================================

def seed_everything(seed=42):
    """
    Seed all random number generators for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Directory Utilities
# ============================================================

def make_directory(path):
    """
    Create directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# ============================================================
# Model Utilities
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained model checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print(f"Path : {checkpoint_path}")

    return checkpoint


# ============================================================
# Result Saving
# ============================================================

def save_json(results, filename):
    """
    Save dictionary as JSON.
    """

    with open(filename, "w") as file:
        json.dump(results, file, indent=4)


def save_csv(results, filename):
    """
    Save dictionary as CSV.
    """

    with open(filename, "w") as file:

        file.write("Metric,Value\n")

        for key, value in results.items():
            file.write(f"{key},{value}\n")


# ============================================================
# Console Printing
# ============================================================

def print_header(title):
    """
    Print section header.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_metrics(metrics):
    """
    Nicely print evaluation metrics.
    """

    print_header("Evaluation Metrics")

    for metric, value in metrics.items():

        if isinstance(value, float):
            print(f"{metric:<30} {value:.4f}")
        else:
            print(f"{metric:<30} {value}")


# ============================================================
# Tensor Utilities
# ============================================================

def to_numpy(tensor):
    """
    Convert tensor to NumPy.
    """

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    return np.asarray(tensor)


# ============================================================
# Device Utility
# ============================================================

def get_device():
    """
    Return available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")