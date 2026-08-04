# ============================================================
# Vision2Drive Benchmark
# ============================================================

import json
import os

from evaluation.metrics import summarize_metrics
from evaluation.visualize import EvaluationVisualizer


class Benchmark:
    """
    Benchmark Behavior Cloning against PPO.
    """

    def __init__(self, output_dir="results"):

        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.visualizer = EvaluationVisualizer(
            output_dir=output_dir
        )

    # ========================================================
    # Compare Metrics
    # ========================================================

    def compare(self, bc_episodes, ppo_episodes):

        bc_metrics = summarize_metrics(bc_episodes)

        ppo_metrics = summarize_metrics(ppo_episodes)

        self.print_report(
            bc_metrics,
            ppo_metrics,
        )

        self.save_results(
            bc_metrics,
            ppo_metrics,
        )

        self.visualizer.benchmark(
            bc_metrics,
            ppo_metrics,
        )

        return bc_metrics, ppo_metrics

    # ========================================================
    # Console Report
    # ========================================================

    def print_report(
        self,
        bc_metrics,
        ppo_metrics,
    ):

        print("\n")
        print("=" * 90)
        print("Vision2Drive Benchmark")
        print("=" * 90)

        print(
            f"{'Metric':<30}"
            f"{'Behavior Cloning':>20}"
            f"{'PPO':>20}"
        )

        print("-" * 90)

        for metric in bc_metrics.keys():

            bc = bc_metrics[metric]
            ppo = ppo_metrics[metric]

            if isinstance(bc, float):

                print(
                    f"{metric:<30}"
                    f"{bc:>20.4f}"
                    f"{ppo:>20.4f}"
                )

            else:

                print(
                    f"{metric:<30}"
                    f"{str(bc):>20}"
                    f"{str(ppo):>20}"
                )

        print("=" * 90)

    # ========================================================
    # Save Results
    # ========================================================

    def save_results(
        self,
        bc_metrics,
        ppo_metrics,
    ):

        results = {

            "Behavior Cloning": bc_metrics,

            "PPO": ppo_metrics,

        }

        filename = os.path.join(
            self.output_dir,
            "benchmark.json",
        )

        with open(filename, "w") as file:

            json.dump(
                results,
                file,
                indent=4,
            )

        print(f"\nBenchmark saved to: {filename}")

    # ========================================================
    # Improvement Report
    # ========================================================

    def improvement(
        self,
        bc_metrics,
        ppo_metrics,
    ):

        print("\n")
        print("=" * 90)
        print("Performance Improvement")
        print("=" * 90)

        for metric in bc_metrics.keys():

            bc = bc_metrics[metric]
            ppo = ppo_metrics[metric]

            if isinstance(bc, (int, float)):

                improvement = ppo - bc

                print(
                    f"{metric:<30}"
                    f"{improvement:+10.4f}"
                )

        print("=" * 90)