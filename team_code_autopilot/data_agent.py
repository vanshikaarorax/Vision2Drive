"""
data_agent.py

Core data collection agent for Vision2Drive-MetaDrive.

The DataAgent acts as a bridge between MetaDrive and the dataset generation
pipeline. It does not control the vehicle or perform any planning. Instead,
it observes the running simulation, gathers information from the ego vehicle
and its sensors, and packages everything into a structured frame suitable for
dataset generation.

Responsibilities
----------------
- Access the MetaDrive environment
- Access the ego vehicle
- Access the engine and navigation system
- Capture one simulation frame
- Return collected information in a structured dictionary

This file intentionally starts as a lightweight skeleton. Data collection
logic is added incrementally in later development stages.

Author: Vanshika Arora
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

class DataAgent:
    """
    Collects information from a running MetaDrive simulation.

    Parameters
    ----------
    env : ScenarioEnv
        Active MetaDrive environment.

    Notes
    -----
    The DataAgent never controls the vehicle.

    PPO Expert
        ↓
    env.step(action)
        ↓
    MetaDrive updates world
        ↓
    DataAgent observes world
    """
    RGB_SENSOR = "rgb"
    LIDAR_SENSOR = "lidar"
    def __init__(self, env) -> None:
        """
        Initialize the DataAgent.

        Parameters
        ----------
        env : ScenarioEnv
            Running MetaDrive environment.
        """

        # ------------------------------------------------------------------
        # Core MetaDrive References
        # ------------------------------------------------------------------

        self.env = env
        self.engine = env.engine

        # Will be populated during reset()
        self.vehicle = None
        self.navigation = None
        self.rgb_camera = None
        # Frame bookkeeping
        self.frame_id = 0
        self.lidar = None
        # Episode metadata
        self.episode_id = 0

# Simulation metadata
        
    def reset(self) -> None:

     self.frame_id = 0
     self.episode_id += 1
     self._connect_engine()
     self._connect_vehicle()
     self._connect_navigation()
     self._connect_rgb_camera()
     self._connect_lidar()
    

    def capture_frame(self) -> Dict[str, Any]:
     """
    Capture one simulation frame.

    Part 3:
        - Vehicle state only
    """

     self._validate_connections()

     frame = self._build_frame()

     self.frame_id += 1

     return frame
    def _connect_vehicle(self) -> None:
      """Connect to the active ego vehicle."""
      self.vehicle = self.env.agent


    def _connect_navigation(self) -> None:
      """Connect to the ego vehicle's navigation module."""
      self.navigation = self.vehicle.navigation


    def _connect_engine(self) -> None: 
       """Refresh engine reference."""
       self.engine = self.env.engine

    def get_vehicle_state(self) -> Dict[str, Any]:
     """
    Collect the current state of the ego vehicle.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the vehicle's current kinematic state.
    """

     vehicle = self.vehicle

     state = {
        "position": vehicle.position.tolist(),
        "heading": float(vehicle.heading_theta),
        "speed": float(vehicle.speed),
        "velocity": vehicle.velocity.tolist(),
        "steering": float(vehicle.steering),
        "throttle": float(vehicle.throttle_brake),
        "brake": float(vehicle.brake),
    }

     return state
    
    def _connect_rgb_camera(self) -> None:
      """
    Connect to the front RGB camera attached to the ego vehicle.
    """

      self.rgb_camera = self.vehicle.image_sensors[self.RGB_SENSOR]
    

    def capture_rgb(self):
     """
    Capture one RGB frame.

    Returns
    -------
    np.ndarray
        RGB image.
    """

     return self.rgb_camera.perceive()
    
    def _connect_lidar(self) -> None:
     """
    Connect to the ego vehicle's LiDAR sensor.
    """

     self.lidar = self.vehicle.lidar

    def capture_lidar(self):
     """
    Capture one LiDAR frame.

    Returns
    -------
    np.ndarray
        LiDAR point cloud for the current simulation step.
    """

     return self.lidar.perceive()
    

    def get_navigation_state(self) -> Dict[str, Any]:
     """
    Collect the current navigation information.

    Returns
    -------
    Dict[str, Any]
        Route information for the current timestep.
    """

     navigation = self.navigation

     nav_state = {

        "current_lane": navigation.current_lane,

        "current_checkpoint": navigation.current_checkpoint,

        "target_checkpoint": navigation.target_checkpoint,

        "route_completion": navigation.route_completion,

        "next_waypoint": navigation.next_ref_lanes,

    }

     return nav_state
    
    def _build_frame(self) -> Dict[str, Any]:
     """
    Build a complete synchronized dataset frame.

    Returns
    -------
    Dict[str, Any]
        Complete information for one simulation timestep.
    """

     frame = {

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------
        "metadata": self.get_metadata(),

        # ---------------------------------------------------------
        # Ego vehicle
        # ---------------------------------------------------------
        "vehicle_state": self.get_vehicle_state(),

        # ---------------------------------------------------------
        # Navigation
        # ---------------------------------------------------------
        "navigation": self.get_navigation_state(),

        # ---------------------------------------------------------
        # Sensors
        # ---------------------------------------------------------
        "rgb": self.capture_rgb(),

        "lidar": self.capture_lidar(),


    }

     return frame
    

    def get_metadata(self) -> Dict[str, Any]:
     """
    Collect metadata describing the current frame.
    """

     metadata = {

        "episode_id": self.episode_id,

        "frame_id": self.frame_id,

        "timestamp": self.engine.global_clock,

        "scenario_seed": self.env.current_seed,

    }

     return metadata
    
    def _validate_connections(self) -> None:
      """
    Validate that all required MetaDrive objects are available before
    collecting a frame.

    Raises
    ------
    RuntimeError
        If any required object is missing.
    """

      if self.engine is None:
        raise RuntimeError("MetaDrive engine is not connected.")

      if self.vehicle is None:
        raise RuntimeError("Ego vehicle is not connected.")

      if self.navigation is None:
        raise RuntimeError("Navigation module is not connected.")

      if self.rgb_camera is None:
        raise RuntimeError("RGB camera is not connected.")

      if self.lidar is None:
        raise RuntimeError("LiDAR sensor is not connected.")
      

    
    def is_ready(self) -> bool:
     """
    Check whether all required references are available.
    """

     return (
        self.engine is not None
        and self.vehicle is not None
        and self.navigation is not None
        and self.rgb_camera is not None
        and self.lidar is not None
    )
    