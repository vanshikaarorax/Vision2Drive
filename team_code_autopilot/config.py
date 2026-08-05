"""
config.py

Central configuration file for the Vision2Drive MetaDrive
dataset generation pipeline.

This module stores all configurable parameters used across
the project, including:

- MetaDrive environment settings
- Dataset generation parameters
- Sensor configuration
- Output paths

Author: Vanshika
"""

from pathlib import Path

from metadrive.metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.metadrive.policy.expert_policy import ExpertPolicy

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "output"


# ==========================================================
# Dataset Configuration
# ==========================================================

NUM_EPISODES = 100

DATASET_NAME = "Vision2Drive"

FRAME_SKIP = 1


# ==========================================================
# RGB Camera Configuration
# ==========================================================

RGB_WIDTH = 1280
RGB_HEIGHT = 720
RGB_FOV = 70


# ==========================================================
# LiDAR Configuration
# ==========================================================

LIDAR_NUM_LASERS = 64
LIDAR_DISTANCE = 100
LIDAR_HORIZONTAL_RESOLUTION = 1800


# ==========================================================
# MetaDrive Environment Configuration
# ==========================================================

METADRIVE_CONFIG = {

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    "use_render": False,

    "manual_control": False,

    # Built-in MetaDrive expert driver
    "agent_policy": ExpertPolicy,

    "traffic_density": 0.1,

    "num_scenarios": NUM_EPISODES,

    "random_agent_model": False,

    "random_lane_width": False,

    "random_lane_num": False,

    # ------------------------------------------------------
    # Camera Sensors
    # ------------------------------------------------------

    "image_observation": True,

    "sensors": {

        "rgb": (
            RGBCamera,
            RGB_WIDTH,
            RGB_HEIGHT,
        ),

    },

    # ------------------------------------------------------
    # Vehicle
    # ------------------------------------------------------

    "vehicle_config": {

        # RGB observation comes from the registered camera
        "image_source": "rgb",

        "lidar": {

            "num_lasers": LIDAR_NUM_LASERS,

            "distance": LIDAR_DISTANCE,

            "num_others": 0,

        },

    },

}


# ==========================================================
# Dataset Generator Configuration
# ==========================================================

DATASET_CONFIG = {

    "num_episodes": NUM_EPISODES,

    "output_dir": OUTPUT_DIR

}