"""
generate_dataset.py

Runs the complete Vision2Drive dataset generation pipeline.

Responsibilities
----------------
1. Create the MetaDrive environment.
2. Load the expert PPO policy.
3. Collect synchronized sensor data using DataAgent.
4. Save the collected dataset using Writer.

Author: Vanshika
"""

from typing import Optional

from metadrive.envs.scenario_env import ScenarioEnv
import time
from config import DATASET_CONFIG
from data_agent import DataAgent
from writer import Writer
from stable_baselines3 import PPO
class DatasetGenerator:
    """
    Orchestrates the complete dataset generation pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize the dataset generator.
        """

        # Configuration
        self.config = DATASET_CONFIG

        # Core components
        self.env: Optional[ScenarioEnv] = None
        self.expert = None
        self.agent: Optional[DataAgent] = None
        self.writer: Optional[Writer] = None
        # Statistics
        self.total_frames = 0
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
     self._build_expert()

    # Generate dataset
     for episode_id in range(self.config["num_episodes"]):

        
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
     self.env = ScenarioEnv(self.config)

    # Create data collection agent
     self.agent = DataAgent(self.env)

    # Create dataset writer
     self.writer = Writer()

    def _build_expert(self) -> None:
        """
    Load the trained PPO expert policy.
    """

        self.expert = PPO.load(
         self.config["expert_checkpoint"]
    )

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

        # Expert action
        action = self._predict_action(observation)

        # Advance simulation
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Capture synchronized frame
        frame = self.agent.capture_frame()

        # Save frame
        self.writer.save(frame)

        # Episode finished?
        done = terminated or truncated

    # Finalize episode
     self.writer.finalize_episode()
     self.total_frames += self.writer.frame_count
    # ==========================================================
    # Cleanup
    # ==========================================================

    def _cleanup(self) -> None:
        """
        Release resources before exiting.
        """
        raise NotImplementedError
    
    def _predict_action(self, observation):
     """
    Predict the next action using the expert policy.
    """

     action, _ = self.expert.predict(
        observation,
        deterministic=True
    )

     return action


def main() -> None:
    """
    Entry point for dataset generation.
    """

    generator = DatasetGenerator()
    generator.run()


if __name__ == "__main__":
    main()