# ============================================================
# Vision2Drive Inference
# ============================================================

import torch

from models.model import Vision2Drive
from evaluation.utils import load_checkpoint


class Vision2DriveInference:
    """
    Runs inference using a trained Vision2Drive model.
    """

    def __init__(self, checkpoint_path, device):

        self.device = device

        self.model = Vision2Drive().to(device)

        load_checkpoint(
            self.model,
            checkpoint_path,
            device,
        )

    # ========================================================
    # Observation Processing
    # ========================================================

    def prepare_inputs(self, observation):
        """
        Convert environment observation into model inputs.
        """

        rgb = observation["rgb"].unsqueeze(0).to(self.device)

        lidar = observation["lidar"].unsqueeze(0).to(self.device)

        vehicle_state = observation["vehicle_state"].unsqueeze(0).to(self.device)

        navigation = observation["navigation"].unsqueeze(0).to(self.device)

        return rgb, lidar, vehicle_state, navigation

    # ========================================================
    # Forward Pass
    # ========================================================

    @torch.no_grad()
    def predict(self, observation):
        """
        Predict driving action.
        """

        rgb, lidar, vehicle_state, navigation = self.prepare_inputs(
            observation
        )

        outputs = self.model(
            rgb,
            lidar,
            vehicle_state,
            navigation,
        )

        steering = outputs["steering"].item()
        throttle = outputs["throttle"].item()
        brake = outputs["brake"].item()

        action = [
            steering,
            throttle,
            brake,
        ]

        return action

    # ========================================================
    # Batch Inference
    # ========================================================

    @torch.no_grad()
    def predict_batch(self, observations):
        """
        Predict actions for multiple observations.
        """

        actions = []

        for observation in observations:

            actions.append(
                self.predict(observation)
            )

        return actions

    # ========================================================
    # Model Information
    # ========================================================

    def summary(self):

        total_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

        print("=" * 60)
        print("Vision2Drive Inference")
        print("=" * 60)

        print(f"Device               : {self.device}")
        print(f"Total Parameters     : {total_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")