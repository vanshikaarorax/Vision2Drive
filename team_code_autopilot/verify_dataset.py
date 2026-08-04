"""
verify_dataset.py

Verifies the integrity of the generated Vision2Drive dataset.

Checks include:
    - Dataset structure
    - Episode structure
    - Frame consistency
    - Missing files
    - Corrupted files

Author: Vanshika
"""

from pathlib import Path
from typing import Dict

from config import OUTPUT_DIR
import json

class DatasetVerifier:
    """
    Verifies the integrity of a generated Vision2Drive dataset.
    """

    def __init__(self, dataset_root: Path = OUTPUT_DIR) -> None:
        """
        Initialize the dataset verifier.

        Args:
            dataset_root: Root directory of the generated dataset.
        """

        self.dataset_root = Path(dataset_root)

        # Statistics
        self.total_episodes = 0
        self.total_frames = 0

        self.missing_files = 0
        self.corrupted_files = 0
        self.failed_episodes = 0

        # Results
        self.report: Dict = {}

    # ==========================================================
    # Public API
    # ==========================================================

    def verify(self) -> None:
        """
        Verify the complete dataset.
        """ 
        self._verify_dataset()

        for episode_dir in self.episode_dirs:

         try:

            self._verify_episode(episode_dir)

         except Exception as error:

            self.failed_episodes += 1

            print(
                f"Failed: {episode_dir.name} ({error})"
            )

        self._generate_report()

        self._print_summary()
        

    # ==========================================================
    # Dataset Verification
    # ==========================================================

    def _verify_dataset(self) -> None:
        """
        Verify dataset-level structure.
        """

        
    # ------------------------------------------------------
    # Dataset directory exists
    # ------------------------------------------------------

        if not self.dataset_root.exists():
          raise FileNotFoundError(
            f"Dataset not found: {self.dataset_root}"
        )

    # ------------------------------------------------------
    # Dataset manifest exists
    # ------------------------------------------------------

        manifest_path = self.dataset_root / "manifest.json"

        if not manifest_path.exists():
          raise FileNotFoundError(
            "Dataset manifest.json is missing."
        )

    # ------------------------------------------------------
    # Validate manifest.json
    # ------------------------------------------------------

        try:

         with open(manifest_path, "r") as file:

            json.load(file)

        except Exception as error:

          raise ValueError(
            f"Invalid manifest.json: {error}"
        )

    # ------------------------------------------------------
    # Discover episode folders
    # ------------------------------------------------------

        episode_dirs = sorted(

           directory

           for directory in self.dataset_root.iterdir()

           if directory.is_dir()
  
           and directory.name.startswith("episode_")

    )

        self.total_episodes = len(episode_dirs)

        if self.total_episodes == 0:

          raise ValueError(
            "No episode folders found."
        )

    # Store for later stages
        self.episode_dirs = episode_dirs

    # ==========================================================
    # Episode Verification
    # ==========================================================

    def _verify_episode(self, episode_dir: Path) -> None:
     """
    Verify a single episode.
    """

    # ------------------------------------------------------
    # Episode manifest
    # ------------------------------------------------------

     manifest = episode_dir / "manifest.json"

     if not manifest.exists():
        raise FileNotFoundError(
            "Missing episode manifest."
        )

     try:

        with open(manifest, "r") as file:

            json.load(file)

     except Exception as error:

        raise ValueError(
            f"Invalid manifest: {error}"
        )

    # ------------------------------------------------------
    # Required directories
    # ------------------------------------------------------

     metadata_dir = episode_dir / "metadata"
     rgb_dir = episode_dir / "rgb"
     lidar_dir = episode_dir / "lidar"

     required_dirs = (
        metadata_dir,
        rgb_dir,
        lidar_dir,
    )

     for directory in required_dirs:

        if not directory.exists():

            raise FileNotFoundError(
                f"Missing directory: {directory.name}"
            )

    # ------------------------------------------------------
    # Collect frame filenames
    # ------------------------------------------------------

     metadata_files = sorted(
        metadata_dir.glob("*.json")
    )

     rgb_files = sorted(
        rgb_dir.glob("*.png")
    )

     lidar_files = sorted(
        lidar_dir.glob("*.npy")
    )

     frame_count = len(metadata_files)

    # ------------------------------------------------------
    # Counts must match
    # ------------------------------------------------------

     if not (
        len(metadata_files)
        == len(rgb_files)
        == len(lidar_files)
    ):

        raise ValueError(
            "Frame count mismatch."
        )

     self.total_frames += frame_count

    # ------------------------------------------------------
    # Verify every frame
    # ------------------------------------------------------

     for metadata in metadata_files:

        frame_id = metadata.stem

        self._verify_frame(
            episode_dir,
            frame_id,
        )
    # ==========================================================
    # Frame Verification
    # ==========================================================

    def _verify_frame(self, episode_dir: Path, frame_id: str) -> None:
        """
        Verify a single frame.
        """

        metadata_path = (
        episode_dir
        / "metadata"
        / f"{frame_id}.json"
    )

        rgb_path = (
        episode_dir
        / "rgb"
        / f"{frame_id}.png"
    )

        lidar_path = (
        episode_dir
        / "lidar"
        / f"{frame_id}.npy"
    )

    # ------------------------------------------------------
    # Files exist
    # ------------------------------------------------------

        paths = (
        metadata_path,
        rgb_path,
        lidar_path,
    )

        for path in paths:

          if not path.exists():

            self.missing_files += 1

            raise FileNotFoundError(
                f"Missing file: {path.name}"
            )

    # ------------------------------------------------------
    # Verify JSON
    # ------------------------------------------------------

        try:

          with open(metadata_path, "r") as file:

            json.load(file)

        except Exception:

          self.corrupted_files += 1

          raise

    # ------------------------------------------------------
    # Verify PNG
    # ------------------------------------------------------

        try:

          with Image.open(rgb_path) as image:

            image.verify()

        except Exception:

          self.corrupted_files += 1

          raise

    # ------------------------------------------------------
    # Verify NumPy
    # ------------------------------------------------------

        try:

          np.load(
            lidar_path,
            allow_pickle=False,
        )

        except Exception:

          self.corrupted_files += 1

        raise

    # ==========================================================
    # Reporting
    # ==========================================================

    def _generate_report(self) -> None:
        """
        Generate verification summary.
        """

        successful_episodes = (
        self.total_episodes
        - self.failed_episodes
    )

        status = (
         "PASS"
         if (
            self.failed_episodes == 0
            and self.missing_files == 0
            and self.corrupted_files == 0
        )
        else "FAIL"
    )

        self.report = {

        "status": status,

        "total_episodes": self.total_episodes,

        "successful_episodes": successful_episodes,

        "failed_episodes": self.failed_episodes,

        "total_frames": self.total_frames,

        "missing_files": self.missing_files,

        "corrupted_files": self.corrupted_files,

    }

    def _print_summary(self) -> None:
        """
        Print verification summary.
        """

        raise NotImplementedError


def main() -> None:
    """
    Entry point.
    """

    verifier = DatasetVerifier()

    verifier.verify()


if __name__ == "__main__":
    main()