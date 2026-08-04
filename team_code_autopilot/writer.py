"""
writer.py

Dataset writer for Vision2Drive-MetaDrive.

The Writer is responsible for persisting synchronized frames collected by the
DataAgent onto disk. It does not communicate with MetaDrive directly. Instead,
it receives a structured frame dictionary and saves each component in the
appropriate format.

Responsibilities
----------------
- Create dataset directories
- Save synchronized simulation frames
- Organize data by episode
- Persist metadata, RGB images, and LiDAR point clouds

Author: Vanshika Arora
"""

from __future__ import annotations
from PIL import Image
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np
class Writer:
    """
    Save synchronized simulation frames to disk.

    Parameters
    ----------
    output_dir : str | Path, optional
        Root directory where the dataset will be written.
        Default is "output".
    """

    def __init__(self, output_dir: str | Path = "output") -> None:
        """
        Initialize the dataset writer.
        """

        # Root output directory
        self.output_dir = Path(output_dir)

        # Current episode information
        self.current_episode = None
        self.episode_path = None

        # Create the dataset root if it does not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Subdirectories for the current episode
        self.metadata_dir = None
        self.rgb_dir = None
        self.lidar_dir = None
        self.frame_count = 0
    # ==========================================================
    # Public API
    # ==========================================================

    def reset(self, episode_id: int) -> None:
        """
        Prepare the writer for a new episode.

        Parameters
        ----------
        episode_id : int
            Episode currently being generated.
        """

        self.current_episode = episode_id

        self._create_episode_folder()
        self.frame_count = 0
        self.frame_count += 1

    def save(self, frame: Dict[str, Any]) -> None:
        """
        Save one synchronized frame.

        Parameters
        ----------
        frame : Dict[str, Any]
            Frame returned by DataAgent.
        """

        raise NotImplementedError(
            "Frame saving will be implemented in later parts."
        )

    # ==========================================================
    # Private Helpers
    # ==========================================================

    def _create_episode_folder(self) -> None:
        """
        Create the directory structure for the current episode.
        """

        raise NotImplementedError(
            "Episode folder creation will be implemented in later parts."
        )
    
    def _create_episode_folder(self) -> None:
     """
    Create the directory structure for the current episode.
    """

    # Episode folder
     self.episode_path = (
        self.output_dir /
        f"episode_{self.current_episode:06d}"
    )

    # Sensor folders
     self.metadata_dir = self.episode_path / "metadata"
     self.rgb_dir = self.episode_path / "rgb"
     self.lidar_dir = self.episode_path / "lidar"

    # Create directories
     self.metadata_dir.mkdir(parents=True, exist_ok=True)
     self.rgb_dir.mkdir(parents=True, exist_ok=True)
     self.lidar_dir.mkdir(parents=True, exist_ok=True)

    def _frame_stem(self, frame_id: int) -> str:
     """
    Return the zero-padded filename stem for a frame.
    """

     return f"{frame_id:06d}"
    
    def _save_metadata(self, frame: Dict[str, Any]) -> None:
     """
    Save metadata for one frame as a JSON file.

    Parameters
    ----------
    frame : Dict[str, Any]
        Frame collected by the DataAgent.
    """

     metadata = {

        "metadata": frame["metadata"],

        "vehicle_state": frame["vehicle_state"],

        "navigation": frame["navigation"],

    }

     frame_id = metadata["metadata"]["frame_id"]

     save_path = (
        self.metadata_dir /
        f"{self._frame_stem(frame_id)}.json"
    )

     with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


    def _save_rgb(self, frame: Dict[str, Any]) -> None:
     """
    Save the RGB image for one frame.

    Parameters
    ----------
    frame : Dict[str, Any]
        Frame collected by the DataAgent.
    """

     rgb = frame["rgb"]

     frame_id = frame["metadata"]["frame_id"]

     save_path = (
        self.rgb_dir /
        f"{self._frame_stem(frame_id)}.png"
    )

     image = Image.fromarray(rgb)

     image.save(save_path)


    def _save_lidar(self, frame: Dict[str, Any]) -> None:
     """
    Save the LiDAR point cloud for one frame.

    Parameters
    ----------
    frame : Dict[str, Any]
        Frame collected by the DataAgent.
    """

     lidar = frame["lidar"]

     frame_id = frame["metadata"]["frame_id"]

     save_path = (
        self.lidar_dir /
        f"{self._frame_stem(frame_id)}.npy"
    )

     np.save(save_path, lidar)


    def save(self, frame: Dict[str, Any]) -> None:
     """
    Save one synchronized simulation frame.
    """

     self._validate_writer()

     self._validate_frame(frame)

     self._save_metadata(frame)

     self._save_rgb(frame)

     self._save_lidar(frame)

    def _validate_writer(self) -> None:
     """
    Validate that the writer has been initialized.
    """

     if self.current_episode is None:
        raise RuntimeError(
            "Writer has not been initialized. "
            "Call writer.reset() before saving frames."
        )

     if self.episode_path is None:
        raise RuntimeError("Episode directory has not been created.")

     if self.metadata_dir is None:
        raise RuntimeError("Metadata directory is missing.")

     if self.rgb_dir is None:
        raise RuntimeError("RGB directory is missing.")

     

    def _validate_frame(self, frame: Dict[str, Any]) -> None:
     """
    Validate the frame before writing it to disk.
    """

     required_keys = {

        "metadata",

        "vehicle_state",

        "navigation",

        "rgb",

        "lidar",

    }

     missing = required_keys - frame.keys()

     if missing:
        raise KeyError(
            f"Frame is missing required fields: {sorted(missing)}"
        )
     

     def _save_manifest(self, frame: Dict[str, Any]) -> None:
      """
    Save or update the episode manifest.
    """

      manifest = {

        "episode_id": frame["metadata"]["episode_id"],

        "scenario_seed": frame["metadata"]["scenario_seed"],

        "total_frames": self.frame_count,

    }

      save_path = self.episode_path / "manifest.json"

      with open(save_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    def save(self, frame: Dict[str, Any]) -> None:
     """
    Save one synchronized simulation frame.
    """

     self._validate_writer()
     self._validate_frame(frame)

     try:
         self._save_metadata(frame)
         self._save_rgb(frame)
         self._save_lidar(frame)

     except Exception as exc:
        raise RuntimeError(
            "Failed to save frame."
        ) from exc

     self.frame_count += 1
       

    def finalize_episode(self, frame: Dict[str, Any]) -> None:
     """
    Finalize the current episode by writing the manifest.

    Parameters
    ----------
    frame : Dict[str, Any]
        Final frame of the episode.
    """

     self._save_manifest(frame)
