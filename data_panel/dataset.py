"""
dataset.py

PyTorch dataset for the Vision2Drive autonomous driving dataset.

Loads:
    - RGB images
    - LiDAR point clouds
    - Vehicle state
    - Navigation state
    - Driving actions

Returns PyTorch-ready tensors for training.

Author: Vanshika
"""

from pathlib import Path
from typing import Dict, List

import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import OUTPUT_DIR


class Vision2DriveDataset(Dataset):
    """
    PyTorch dataset for Vision2Drive.
    """

def __init__(
        self,
        dataset_root: Path = OUTPUT_DIR,
        transform=None,
    ) -> None:

        self.dataset_root = Path(dataset_root)

        self.transform = transform

        self.samples: List[Dict] = []

        self._index_dataset()

    # =====================================================
    # PyTorch API
    # =====================================================

def __len__(self) -> int:
    """
    Total number of samples.
    """

    return len(
        self.samples
    )

def __getitem__(
    self,
    index: int,
) -> Dict:
    """
    Return one training sample.
    """

    sample = self.samples[index]

    return self._build_sample(
        sample
    )

    # =====================================================
    # Dataset Indexing
    # =====================================================

def _index_dataset(self) -> None:
    """
    Index every frame in the dataset.

    Only file paths are stored.
    Data is loaded lazily inside __getitem__().
    """

    episode_dirs = sorted(

        directory

        for directory in self.dataset_root.iterdir()

        if directory.is_dir()

        and directory.name.startswith("episode_")

    )

    for episode_dir in episode_dirs:

        metadata_dir = episode_dir / "metadata"

        rgb_dir = episode_dir / "rgb"

        lidar_dir = episode_dir / "lidar"

        metadata_files = sorted(
            metadata_dir.glob("*.json")
        )

        for metadata_path in metadata_files:

            frame_id = metadata_path.stem

            sample = {

                "rgb": (
                    rgb_dir /
                    f"{frame_id}.png"
                ),

                "lidar": (
                    lidar_dir /
                    f"{frame_id}.npy"
                ),

                "metadata": metadata_path,

            }

            self.samples.append(sample)

    # =====================================================
    # Loading
    # =====================================================

def _load_rgb(self, path: Path) -> np.ndarray:
    """
    Load RGB image.
    """

    image = cv2.imread(str(path))

    if image is None:

        raise RuntimeError(
            f"Unable to load image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return image

def _load_lidar(self, path: Path) -> np.ndarray:
    """
    Load LiDAR point cloud.
    """

    try:

        lidar = np.load(
            path,
            allow_pickle=False,
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to load LiDAR: {error}"
        )

    return lidar

def _load_metadata(
    self,
    path: Path,
) -> Dict:
    """
    Load metadata JSON.
    """

    with open(path, "r") as file:

        metadata = json.load(file)

    return metadata

    # =====================================================
    # Processing
    # =====================================================

def _create_bev(
    self,
    lidar: np.ndarray,
    grid_size: int = 200,
    area_size: float = 40.0,
) -> np.ndarray:
    """
    Convert LiDAR point cloud to Bird's-Eye View occupancy map.

    Output:
        (1, grid_size, grid_size)
    """

    bev = np.zeros(
        (grid_size, grid_size),
        dtype=np.float32,
    )

    half = area_size / 2.0
    resolution = area_size / grid_size

    for point in lidar:

        x = point[0]
        y = point[1]

        if (
            -half <= x < half
            and
            -half <= y < half
        ):

            row = int((half - y) / resolution)
            col = int((x + half) / resolution)

            bev[row, col] = 1.0

    return bev[np.newaxis, :, :]

def _prepare_state(
    self,
    metadata: Dict,
) -> np.ndarray:
    """
    Build vehicle state vector.
    """

    state = np.array(

        [

            metadata["speed"],

            metadata["steering"],

            metadata["throttle"],

            metadata["brake"],

            metadata["current_lane"],

            metadata["target_lane"],

            metadata["route_completion"],

        ],

        dtype=np.float32,

    )

    return state

def _prepare_action(
    self,
    metadata: Dict,
) -> np.ndarray:
    """
    Driving action labels.
    """

    action = np.array(

        [

            metadata["steering"],

            metadata["throttle"],

            metadata["brake"],

        ],

        dtype=np.float32,

    )

    return action

def _build_sample(
    self,
    sample: Dict,
) -> Dict:
    """
    Load and prepare one training sample.
    """

    image = self._load_rgb(
        sample["rgb"]
    )

    lidar = self._load_lidar(
        sample["lidar"]
    )

    metadata = self._load_metadata(
        sample["metadata"]
    )

    bev = self._create_bev(
        lidar
    )

    state = self._prepare_state(
        metadata
    )

    action = self._prepare_action(
        metadata
    )

    if self.transform is not None:

        image = self.transform(image)

    else:

        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

    bev = torch.from_numpy(
        bev
    ).float()

    state = torch.from_numpy(
        state
    ).float()

    action = torch.from_numpy(
        action
    ).float()

    return {

        "image": image,

        "lidar": bev,

        "state": state,

        "action": action,

    }