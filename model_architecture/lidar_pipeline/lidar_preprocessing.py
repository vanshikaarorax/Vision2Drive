"""
lidar_preprocessing.py

Converts raw LiDAR point clouds into Bird's-Eye View (BEV)
occupancy maps for Vision2Drive.

Input:
    LiDAR Point Cloud
    Shape: (N, 4)
    Format: [x, y, z, intensity]

Output:
    BEV Occupancy Map
    Shape: (1, H, W)

Author: Vanshika
"""

import numpy as np


class LiDARPreprocessor:
  """
    Converts raw LiDAR point clouds into BEV occupancy maps.
    """

  def __init__(
        self,
        x_range=(-40.0, 40.0),
        y_range=(-40.0, 40.0),
        z_range=(-2.5, 2.0),
        resolution=0.4,
    ):
        """
        Initialize BEV generation parameters.

        Args:
            x_range:
                Forward/backward range in metres.

            y_range:
                Left/right range in metres.

            z_range:
                Vertical range used for filtering.

            resolution:
                Size of one BEV pixel in metres.
        """

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.resolution = resolution

        self.width = int(
            (x_range[1] - x_range[0]) / resolution
        )

        self.height = int(
            (y_range[1] - y_range[0]) / resolution
        )

  def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """
        Remove points outside the configured BEV region.

        Args:
            points:
                Raw LiDAR point cloud of shape (N, 4).

        Returns:
            Filtered point cloud.
        """

        mask = (
            (points[:, 0] >= self.x_range[0]) &
            (points[:, 0] <= self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) &
            (points[:, 1] <= self.y_range[1]) &
            (points[:, 2] >= self.z_range[0]) &
            (points[:, 2] <= self.z_range[1])
        )

        return points[mask]

def _project_to_bev(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Project filtered 3D LiDAR points onto the BEV grid.

        Args:
            points:
                Filtered LiDAR point cloud.

        Returns:
            Integer pixel coordinates of shape (N, 2).
        """

        x = (
            (points[:, 0] - self.x_range[0])
            / self.resolution
        ).astype(np.int32)

        y = (
            (points[:, 1] - self.y_range[0])
            / self.resolution
        ).astype(np.int32)

        return np.stack((x, y), axis=1)

def _create_occupancy_grid(
        self,
        pixels: np.ndarray,
    ) -> np.ndarray:
        """
        Create a Bird's-Eye View occupancy grid.

        Args:
            pixels:
                BEV pixel coordinates of shape (N, 2).

        Returns:
            Occupancy grid of shape (1, H, W).
        """

        # Create empty BEV image
        bev = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )

        # Mark occupied cells
        for x, y in pixels:

            if (
                0 <= x < self.width and
                0 <= y < self.height
            ):
                bev[y, x] = 1.0

        # Add channel dimension
        bev = np.expand_dims(bev, axis=0)

        return bev

def preprocess(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Complete LiDAR preprocessing pipeline.

        Args:
            points:
                Raw LiDAR point cloud.

        Returns:
            Bird's-Eye View occupancy map.
        """

        points = self._filter_points(points)

        pixels = self._project_to_bev(points)

        bev = self._create_occupancy_grid(pixels)

        return bev


if __name__ == "__main__":

    print("=" * 60)
    print("Vision2Drive LiDAR Preprocessing Test")
    print("=" * 60)

    processor = LiDARPreprocessor()

    # Generate dummy LiDAR point cloud
    num_points = 5000

    x = np.random.uniform(-50, 50, num_points)
    y = np.random.uniform(-50, 50, num_points)
    z = np.random.uniform(-3, 3, num_points)
    intensity = np.random.uniform(0, 1, num_points)

    point_cloud = np.column_stack(
        (x, y, z, intensity)
    )

    bev = processor.preprocess(point_cloud)

    print(f"Point Cloud Shape : {point_cloud.shape}")
    print(f"BEV Shape         : {bev.shape}")

    occupied = np.count_nonzero(bev)

    print(f"Occupied Cells    : {occupied}")
    print(f"Grid Height       : {processor.height}")
    print(f"Grid Width        : {processor.width}")

    print("\nLiDAR preprocessing completed successfully.")