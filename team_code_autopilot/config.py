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
- Expert policy checkpoint

Author: Vanshika
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "output"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

EXPERT_CHECKPOINT = CHECKPOINT_DIR / "ppo_expert.zip"


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

    # General
    "use_render": False,

    "manual_control": False,

    "traffic_density": 0.1,

    "num_scenarios": NUM_EPISODES,

    "random_agent_model": False,

    "random_lane_width": False,

    "random_lane_num": False,

    # Sensors
    "image_observation": False,

    "vehicle_config": {

        "image_source": "rgb_camera",

        "rgb_camera": {

            "width": RGB_WIDTH,

            "height": RGB_HEIGHT,

            "fov": RGB_FOV,

        },

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

    "output_dir": OUTPUT_DIR,

    "expert_checkpoint": EXPERT_CHECKPOINT,

    **METADRIVE_CONFIG,

}



# Vision2Drive/

# ├── manifest.json                    # Dataset-level summary
# │
# ├── episode_000001/
# │
# ├── episode_000002/
# │
# ├── episode_000003/
# │
# ├── ...
# │
# └── episode_001000/


# episode_000001/

# ├── manifest.json
# │
# ├── metadata/
# │      000000.json
# │      000001.json
# │      000002.json
# │      ...
# │
# ├── rgb/
# │      000000.png
# │      000001.png
# │      000002.png
# │      ...
# │
# ├── lidar/
# │      000000.npy
# │      000001.npy
# │      000002.npy
# │      ...

# For example

# metadata/000143.json

# rgb/000143.png

# lidar/000143.npy

# all belong to the same simulation timestep.

# Vision2Drive/

# │
# ├── manifest.json
# │
# ├── episode_000001/
# │      │
# │      ├── manifest.json
# │      │
# │      ├── metadata/
# │      │      000000.json
# │      │      000001.json
# │      │      ...
# │      │
# │      ├── rgb/
# │      │      000000.png
# │      │      000001.png
# │      │      ...
# │      │
# │      └── lidar/
# │             000000.npy
# │             000001.npy
# │             ...
# │
# ├── episode_000002/
# │
# ├── episode_000003/
# │
# └── ...