# ============================================================
# Episode Recorder
# ============================================================

import numpy as np


class EpisodeRecorder:
    """
    Records complete episode information during evaluation.
    """

    def __init__(self):
        self.reset()

    # ========================================================
    # Reset Episode
    # ========================================================

    def reset(self):

        self.rewards = []

        self.steering = []
        self.throttle = []
        self.brake = []

        self.speeds = []
        self.positions = []
        self.headings = []

        self.lane_deviation = []

        self.collisions = []
        self.offroad = []
        self.success = []

        self.timestamps = []

    # ========================================================
    # Record One Step
    # ========================================================

    def record(
        self,
        reward,
        action,
        info,
        timestamp,
    ):
        """
        Record one environment step.
        """

        steering, throttle, brake = action

        self.rewards.append(reward)

        self.steering.append(float(steering))
        self.throttle.append(float(throttle))
        self.brake.append(float(brake))

        self.speeds.append(info.get("speed", 0.0))

        self.positions.append(
            info.get(
                "position",
                (0.0, 0.0),
            )
        )

        self.headings.append(
            info.get(
                "heading",
                0.0,
            )
        )

        self.lane_deviation.append(
            info.get(
                "lane_deviation",
                0.0,
            )
        )

        self.collisions.append(
            info.get(
                "crash_vehicle",
                False
            )
            or
            info.get(
                "crash_object",
                False
            )
            or
            info.get(
                "crash_building",
                False
            )
        )

        self.offroad.append(
            info.get(
                "out_of_road",
                False,
            )
        )

        self.success.append(
            info.get(
                "arrive_dest",
                False,
            )
        )

        self.timestamps.append(timestamp)

    # ========================================================
    # Episode Statistics
    # ========================================================

    def get_episode(self):
        """
        Return complete episode statistics.
        """

        return {

            "reward": float(np.sum(self.rewards)),

            "average_reward": float(np.mean(self.rewards))
            if self.rewards else 0.0,

            "episode_length": len(self.rewards),

            "success": bool(any(self.success)),

            "collision": bool(any(self.collisions)),

            "offroad": bool(any(self.offroad)),

            "average_speed": float(np.mean(self.speeds))
            if self.speeds else 0.0,

            "max_speed": float(np.max(self.speeds))
            if self.speeds else 0.0,

            "lane_deviation": float(np.mean(self.lane_deviation))
            if self.lane_deviation else 0.0,

            "steering": np.asarray(self.steering),

            "throttle": np.asarray(self.throttle),

            "brake": np.asarray(self.brake),

            "speed": np.asarray(self.speeds),

            "position": np.asarray(self.positions),

            "heading": np.asarray(self.headings),

            "timestamps": np.asarray(self.timestamps)

        }

    # ========================================================
    # Summary
    # ========================================================

    def summary(self):

        episode = self.get_episode()

        print("=" * 60)
        print("Episode Summary")
        print("=" * 60)

        print(f"Episode Length    : {episode['episode_length']}")
        print(f"Total Reward      : {episode['reward']:.2f}")
        print(f"Average Speed     : {episode['average_speed']:.2f}")
        print(f"Maximum Speed     : {episode['max_speed']:.2f}")
        print(f"Lane Deviation    : {episode['lane_deviation']:.4f}")
        print(f"Collision         : {episode['collision']}")
        print(f"Off Road          : {episode['offroad']}")
        print(f"Destination       : {episode['success']}")