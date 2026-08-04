# ============================================================
# Visualization Utilities
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt


class EvaluationVisualizer:
    """
    Creates evaluation figures for Vision2Drive.
    """

    def __init__(self, output_dir="results", figsize=(12, 6), dpi=300):

        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

        os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # Save Figure
    # ========================================================

    def _save(self, filename):

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename),
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close()

    # ========================================================
    # Reward Curve
    # ========================================================

    def reward_curve(self, rewards):

        plt.figure(figsize=self.figsize)

        plt.plot(rewards, linewidth=2)

        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.grid(True)

        self._save("reward_curve.png")

    # ========================================================
    # Speed Distribution
    # ========================================================

    def speed_distribution(self, speeds):

        plt.figure(figsize=self.figsize)

        plt.hist(speeds, bins=30)

        plt.title("Vehicle Speed Distribution")
        plt.xlabel("Speed")
        plt.ylabel("Frequency")

        self._save("speed_distribution.png")

    # ========================================================
    # Steering Distribution
    # ========================================================

    def steering_distribution(self, steering):

        plt.figure(figsize=self.figsize)

        plt.hist(steering, bins=40)

        plt.title("Steering Distribution")
        plt.xlabel("Steering")
        plt.ylabel("Frequency")

        self._save("steering_distribution.png")

    # ========================================================
    # Throttle Distribution
    # ========================================================

    def throttle_distribution(self, throttle):

        plt.figure(figsize=self.figsize)

        plt.hist(throttle, bins=40)

        plt.title("Throttle Distribution")
        plt.xlabel("Throttle")
        plt.ylabel("Frequency")

        self._save("throttle_distribution.png")

    # ========================================================
    # Brake Distribution
    # ========================================================

    def brake_distribution(self, brake):

        plt.figure(figsize=self.figsize)

        plt.hist(brake, bins=40)

        plt.title("Brake Distribution")
        plt.xlabel("Brake")
        plt.ylabel("Frequency")

        self._save("brake_distribution.png")

    # ========================================================
    # Episode Length
    # ========================================================

    def episode_lengths(self, lengths):

        plt.figure(figsize=self.figsize)

        plt.hist(lengths, bins=20)

        plt.title("Episode Length Distribution")
        plt.xlabel("Steps")
        plt.ylabel("Episodes")

        self._save("episode_lengths.png")

    # ========================================================
    # Trajectory Plot
    # ========================================================

    def trajectory(self, positions):

        positions = np.asarray(positions)

        plt.figure(figsize=(8, 8))

        plt.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=2,
        )

        plt.scatter(
            positions[0, 0],
            positions[0, 1],
            s=80,
            marker="o",
            label="Start",
        )

        plt.scatter(
            positions[-1, 0],
            positions[-1, 1],
            s=80,
            marker="X",
            label="End",
        )

        plt.title("Vehicle Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        self._save("trajectory.png")

    # ========================================================
    # Success Rate
    # ========================================================

    def success_rate(self, successes):

        success = np.sum(successes)
        failure = len(successes) - success

        plt.figure(figsize=(6, 6))

        plt.pie(
            [success, failure],
            labels=["Success", "Failure"],
            autopct="%1.1f%%",
            startangle=90,
        )

        plt.title("Episode Success Rate")

        self._save("success_rate.png")

    # ========================================================
    # BC vs PPO
    # ========================================================

    def benchmark(self, bc_metrics, ppo_metrics):

        labels = list(bc_metrics.keys())

        bc_values = list(bc_metrics.values())
        ppo_values = list(ppo_metrics.values())

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(14, 6))

        plt.bar(
            x - width / 2,
            bc_values,
            width,
            label="Behavior Cloning",
        )

        plt.bar(
            x + width / 2,
            ppo_values,
            width,
            label="PPO",
        )

        plt.xticks(
            x,
            labels,
            rotation=30,
            ha="right",
        )

        plt.title("Behavior Cloning vs PPO")
        plt.legend()

        self._save("bc_vs_ppo.png")

    # ========================================================
    # Generate All Figures
    # ========================================================

    def generate(self, episodes, metrics):

        rewards = [e["reward"] for e in episodes]
        lengths = [e["episode_length"] for e in episodes]
        successes = [e["success"] for e in episodes]

        speeds = np.concatenate(
            [e["speed"] for e in episodes]
        )

        steering = np.concatenate(
            [e["steering"] for e in episodes]
        )

        throttle = np.concatenate(
            [e["throttle"] for e in episodes]
        )

        brake = np.concatenate(
            [e["brake"] for e in episodes]
        )

        self.reward_curve(rewards)

        self.speed_distribution(speeds)

        self.steering_distribution(steering)

        self.throttle_distribution(throttle)

        self.brake_distribution(brake)

        self.episode_lengths(lengths)

        self.success_rate(successes)

        self.trajectory(
            episodes[-1]["position"]
        )

        print("=" * 60)
        print("Evaluation Figures Saved")
        print("=" * 60)