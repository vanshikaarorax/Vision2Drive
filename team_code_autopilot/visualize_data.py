"""
visualize_dataset.py

Interactive dataset browser for the Vision2Drive dataset.

Allows inspection of:

    - RGB images
    - Vehicle state
    - Navigation state
    - LiDAR information

Author: Vanshika
"""

from pathlib import Path
from typing import Dict

import cv2
import json
import numpy as np

from config import OUTPUT_DIR


class DatasetVisualizer:
    """
    Interactive Vision2Drive dataset browser.
    """

    def __init__(
        self,
        dataset_root: Path = OUTPUT_DIR,
    ) -> None:

        self.dataset_root = Path(dataset_root)

        self.episode_dirs = []

        self.current_episode = 0

        self.current_frame = 0

        self.frame_count = 0

        self.frame_data: Dict = {}

    # ======================================================
    # Public API
    # ======================================================

    def visualize(self) -> None:
     """
    Launch dataset browser.
    """

     self.episode_dirs = sorted(

        directory

        for directory in self.dataset_root.iterdir()

        if directory.is_dir()

        and directory.name.startswith("episode_")

    )

     if not self.episode_dirs:

        raise RuntimeError(
            "No episodes found."
        )

     self._load_episode()

     while True:

        self._load_frame()

        self._display_frame()

        if not self._handle_keyboard():

            break

     cv2.destroyAllWindows()

    # ======================================================
    # Loading
    # ======================================================

    def _load_episode(self) -> None:
        self.episode_dir = self.episode_dirs[
        self.current_episode
    ]

        metadata_dir = (
        self.episode_dir
        / "metadata"
    )

        self.metadata_files = sorted(
        metadata_dir.glob("*.json")
    )

        self.frame_count = len(
        self.metadata_files
    )

        self.current_frame = 0

    def _load_frame(self) -> None:
     frame_id = (
        self.metadata_files[
            self.current_frame
        ].stem
    )

     metadata_path = (
        self.episode_dir
        / "metadata"
        / f"{frame_id}.json"
    )

     rgb_path = (
        self.episode_dir
        / "rgb"
        / f"{frame_id}.png"
    )

     lidar_path = (
        self.episode_dir
        / "lidar"
        / f"{frame_id}.npy"
    )

     with open(metadata_path) as file:

        metadata = json.load(file)

     rgb = cv2.imread(str(rgb_path))

     lidar = np.load(
        lidar_path,
        allow_pickle=False,
    )

     self.frame_data = {

        "frame_id": frame_id,

        "metadata": metadata,

        "rgb": rgb,

        "lidar": lidar,

    }

    # ======================================================
    # Display
    # ======================================================

    def _display_frame(self) -> None:
        """
    Display current frame.
    """

        image = self.frame_data["rgb"].copy()

        image = self._draw_overlay(image)

        window_title = (

        f"Vision2Drive | "

        f"Episode {self.current_episode + 1}/{len(self.episode_dirs)} | "

        f"Frame {self.current_frame + 1}/{self.frame_count}"

    )

        cv2.imshow(

        window_title,

        image,

    )

    def _draw_overlay(self, image):
        metadata = self.frame_data["metadata"]

        vehicle = metadata["vehicle_state"]

        navigation = metadata["navigation"]

        lines = [

        f"Episode : {self.current_episode + 1}",

        f"Frame   : {self.frame_data['frame_id']}",

        "",

        f"Speed      : {vehicle['speed']:.2f}",

        f"Steering   : {vehicle['steering']:.2f}",

        f"Throttle   : {vehicle['throttle']:.2f}",

        f"Brake      : {vehicle['brake']:.2f}",

        "",

        f"Lane       : {navigation['current_lane']}",

        f"Target     : {navigation['target_lane']}",

        f"Progress   : {navigation['route_completion']:.2f}",

        "",

        f"LiDAR Shape: {self.frame_data['lidar'].shape}"

    ]

        y = 30

        for line in lines:

          cv2.putText(

            image,

            line,

            (20, y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2,

            cv2.LINE_AA,

        )

        y += 28

        return image

    # ======================================================
    # Navigation
    # ======================================================

    def _handle_keyboard(self) -> bool:
        """
    Handle keyboard navigation.

    Controls
    --------
    N : Next frame
    P : Previous frame
    E : Next episode
    B : Previous episode
    Q / ESC : Quit
    """

        key = cv2.waitKey(0) & 0xFF

    # ------------------------------------
    # Quit
    # ------------------------------------

        if key in (ord("q"), 27):
         return False

    # ------------------------------------
    # Next Frame
    # ------------------------------------

        elif key == ord("n"):

         if self.current_frame < self.frame_count - 1:

            self.current_frame += 1

    # ------------------------------------
    # Previous Frame
    # ------------------------------------

        elif key == ord("p"):

         if self.current_frame > 0:

            self.current_frame -= 1

    # ------------------------------------
    # Next Episode
    # ------------------------------------

        elif key == ord("e"):

         if self.current_episode < len(self.episode_dirs) - 1:

            self.current_episode += 1

            self._load_episode()

    # ------------------------------------
    # Previous Episode
    # ------------------------------------

        elif key == ord("b"):

         if self.current_episode > 0:

            self.current_episode -= 1

            self._load_episode()

        return True


def main():

    viewer = DatasetVisualizer()

    viewer.visualize()


if __name__ == "__main__":

    main()