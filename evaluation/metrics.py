# ============================================================
# Evaluation Metrics
# ============================================================

import numpy as np


def compute_average_reward(rewards):
    """Average reward across episodes."""
    return float(np.mean(rewards))


def compute_success_rate(successes):
    """Percentage of successful episodes."""
    return float(np.mean(successes) * 100)


def compute_collision_rate(collisions):
    """Average collision rate."""
    return float(np.mean(collisions))


def compute_offroad_rate(offroad):
    """Average off-road rate."""
    return float(np.mean(offroad))


def compute_average_speed(speeds):
    """Average vehicle speed."""
    return float(np.mean(speeds))


def compute_average_episode_length(lengths):
    """Average episode length."""
    return float(np.mean(lengths))


def compute_lane_deviation(deviations):
    """Average lane deviation."""
    return float(np.mean(deviations))


def compute_steering_smoothness(steering):
    """
    Average change in steering angle.
    Lower is smoother.
    """

    steering = np.asarray(steering)

    if len(steering) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(steering))))


def compute_throttle_smoothness(throttle):
    """
    Average throttle change.
    Lower is smoother.
    """

    throttle = np.asarray(throttle)

    if len(throttle) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(throttle))))


def compute_brake_usage(brakes):
    """Average brake intensity."""
    return float(np.mean(brakes))


def summarize_metrics(episodes):
    """
    Compute all evaluation metrics.

    Args:
        episodes: List of episode dictionaries collected by recorder.py

    Returns:
        Dictionary containing evaluation statistics.
    """

    rewards = [e["reward"] for e in episodes]
    successes = [e["success"] for e in episodes]
    collisions = [e["collision"] for e in episodes]
    offroad = [e["offroad"] for e in episodes]
    speeds = [e["average_speed"] for e in episodes]
    lengths = [e["episode_length"] for e in episodes]
    deviations = [e["lane_deviation"] for e in episodes]

    steering = np.concatenate([e["steering"] for e in episodes])
    throttle = np.concatenate([e["throttle"] for e in episodes])
    brakes = np.concatenate([e["brake"] for e in episodes])

    metrics = {
        "Average Reward": compute_average_reward(rewards),
        "Success Rate (%)": compute_success_rate(successes),
        "Collision Rate": compute_collision_rate(collisions),
        "Off-road Rate": compute_offroad_rate(offroad),
        "Average Speed": compute_average_speed(speeds),
        "Average Episode Length": compute_average_episode_length(lengths),
        "Lane Deviation": compute_lane_deviation(deviations),
        "Steering Smoothness": compute_steering_smoothness(steering),
        "Throttle Smoothness": compute_throttle_smoothness(throttle),
        "Average Brake Usage": compute_brake_usage(brakes),
    }

    return metrics