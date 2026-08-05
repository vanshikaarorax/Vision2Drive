"""
generate_dataset.py

Runs the complete Vision2Drive dataset generation pipeline.

Responsibilities
----------------
1. Create the MetaDrive environment.
2. Use MetaDrive's built-in ExpertPolicy to drive the ego vehicle.
3. Collect synchronized sensor data using DataAgent.
4. Save the collected dataset using Writer.

Author: Vanshika
"""

from typing import Optional
import time

from metadrive.metadrive.envs.metadrive_env import MetaDriveEnv

from .config import METADRIVE_CONFIG, DATASET_CONFIG
from .data_agent import DataAgent
from .writer import Writer


class DatasetGenerator:
    """
    Orchestrates the complete dataset generation pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize the dataset generator.
        """

        # Configuration
        self.env_config = METADRIVE_CONFIG
        self.dataset_config = DATASET_CONFIG

        # Core components
        self.env: Optional[MetaDriveEnv] = None
        self.agent: Optional[DataAgent] = None
        self.writer: Optional[Writer] = None

        # Statistics
        self.total_frames = 0
        self.failed_episodes = 0
        self.start_time = 0.0

    # ==========================================================
    # Public API
    # ==========================================================

    def run(self) -> None:
        """
        Run the complete dataset generation pipeline.
        """

        self.start_time = time.time()

        # Initialize components
        self._build_environment()

        # Generate dataset
        for episode_id in range(self.dataset_config["num_episodes"]):

            try:

                self._generate_episode(episode_id)

                self.total_frames += self.writer.frame_count

            except Exception as error:

                self.failed_episodes += 1

                print(
                    f"Episode {episode_id} failed: {error}"
                )

                continue

            elapsed = time.time() - self.start_time

            print(
                f"[{episode_id + 1}/{self.config['num_episodes']}] "
                f"Frames: {self.total_frames} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Release resources
        self._cleanup()

    # ==========================================================
    # Build Helpers
    # ==========================================================

    def _build_environment(self) -> None:
        """
        Create the MetaDrive environment and initialize
        the dataset components.
        """

        # Create simulation environment
        self.env = MetaDriveEnv(self.env_config)

        # Create data collection agent
        self.agent = DataAgent(self.env)

        # Create dataset writer
        self.writer = Writer(self.dataset_config["output_dir"])

    # ==========================================================
    # Dataset Generation
    # ==========================================================

    def _generate_episode(self, episode_id: int) -> None:
        """
        Generate one complete driving episode.
        """

        # Reset simulation
        observation, _ = self.env.reset()

        # Reset helper classes
        self.agent.reset()

        self.writer.reset(episode_id)

        done = False

        while not done:

            # Vehicle is controlled automatically by
            # MetaDrive's ExpertPolicy.
            observation, reward, terminated, truncated, info = (
                self.env.step([0.0, 0.0])
            )

            # Capture synchronized frame
            frame = self.agent.capture_frame()

            # Save frame
            self.writer.save(frame)

            # Episode finished?
            done = terminated or truncated

        # Finalize episode
        self.writer.finalize_episode()

    # ==========================================================
    # Cleanup
    # ==========================================================

    def _cleanup(self) -> None:
        """
        Release resources before exiting.
        """

        if self.env is not None:
            self.env.close()


def main() -> None:
    """
    Entry point for dataset generation.
    """

    generator = DatasetGenerator()
    generator.run()


if __name__ == "__main__":
    main()