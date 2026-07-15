"""
Dataset loader for ultrasound vein detection
Handles video frame extraction and annotation parsing from Sample_Data
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import logging
from collections import defaultdict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class UltrasoundVeinDataset(Dataset):
    """Dataset for ultrasound vein detection from video frames"""

    def __init__(
        self,
        video_paths: List[Path],
        annotation_type: str = 'full',  # 'full' or 'simple'
        frame_stride: int = 5,
        image_size: int = 512,
        augment: bool = False,
    ):
        """
        Args:
            video_paths: List of paths to video files
            annotation_type: 'simple' (fascia + veins) or 'full' (fascia + veins + N1/N2/N3)
            frame_stride: Sample every nth frame
            image_size: Target image size
            augment: Apply data augmentation
        """
        self.video_paths = video_paths
        self.annotation_type = annotation_type
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.augment = augment

        self.frames = []
        self.annotations = []
        self.video_indices = []

        self._load_data()

    def _load_data(self):
        """Load frames from videos"""
        logger.info(f"Loading {len(self.video_paths)} videos...")

        for video_idx, video_path in enumerate(self.video_paths):
            logger.info(f"Loading video {video_idx + 1}/{len(self.video_paths)}: {video_path.name}")

            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % self.frame_stride == 0:
                    # Normalize frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frame = frame.astype(np.float32) / 255.0

                    self.frames.append(frame)
                    self.video_indices.append(video_idx)

                    # Create mock annotation (will be enhanced with actual parsing)
                    annotation = self._create_mock_annotation(frame)
                    self.annotations.append(annotation)

                frame_num += 1

            cap.release()

        logger.info(f"Loaded {len(self.frames)} frames from {len(self.video_paths)} videos")

    def _create_mock_annotation(self, frame: np.ndarray) -> Dict:
        """Create mock annotation for frame (placeholder for actual annotation parsing)"""
        h, w = frame.shape[:2]

        # Mock fascia detection (horizontal line in middle)
        fascia_y = h // 2 + np.random.randint(-50, 50)

        # Mock vein detections (random circles above and below fascia)
        veins = []

        # Deep veins (below fascia, N1)
        num_deep = np.random.randint(1, 3)
        for _ in range(num_deep):
            vein = {
                'x': np.random.randint(w // 4, 3 * w // 4),
                'y': np.random.randint(fascia_y + 20, h - 20),
                'radius': np.random.randint(10, 25),
                'classification': 'N1'
            }
            veins.append(vein)

        # Veins at fascia (N2)
        num_fascia = np.random.randint(0, 2)
        for _ in range(num_fascia):
            vein = {
                'x': np.random.randint(w // 4, 3 * w // 4),
                'y': fascia_y + np.random.randint(-15, 15),
                'radius': np.random.randint(10, 20),
                'classification': 'N2'
            }
            veins.append(vein)

        # Superficial veins (above fascia, N3)
        num_superficial = np.random.randint(1, 3)
        for _ in range(num_superficial):
            vein = {
                'x': np.random.randint(w // 4, 3 * w // 4),
                'y': np.random.randint(20, fascia_y - 20),
                'radius': np.random.randint(8, 20),
                'classification': 'N3'
            }
            veins.append(vein)

        return {
            'fascia_y': fascia_y,
            'veins': veins,
            'image_size': (h, w)
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get frame and annotation at index"""
        frame = self.frames[idx]
        annotation = self.annotations[idx]

        # Convert to tensor
        image = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)

        # Create segmentation mask
        mask = self._create_segmentation_mask(annotation)
        mask = torch.from_numpy(mask).long()

        # Create classification mask (N1/N2/N3)
        classification_mask = self._create_classification_mask(annotation)
        classification_mask = torch.from_numpy(classification_mask).long()

        # Create fascia mask
        fascia_mask = self._create_fascia_mask(annotation)
        fascia_mask = torch.from_numpy(fascia_mask).long()

        return {
            'image': image,
            'vein_mask': mask,
            'classification_mask': classification_mask,
            'fascia_mask': fascia_mask,
            'video_idx': self.video_indices[idx]
        }

    def _create_segmentation_mask(self, annotation: Dict) -> np.ndarray:
        """Create vein segmentation mask"""
        h, w = annotation['image_size']
        mask = np.zeros((h, w), dtype=np.uint8)

        for vein in annotation['veins']:
            cv2.circle(
                mask,
                (vein['x'], vein['y']),
                vein['radius'],
                1,  # vein class
                -1  # filled
            )

        return mask

    def _create_classification_mask(self, annotation: Dict) -> np.ndarray:
        """Create vein classification mask (N1=0, N2=1, N3=2)"""
        h, w = annotation['image_size']
        mask = np.zeros((h, w), dtype=np.uint8)

        class_map = {'N1': 0, 'N2': 1, 'N3': 2}

        for vein in annotation['veins']:
            class_id = class_map.get(vein['classification'], 0)
            cv2.circle(
                mask,
                (vein['x'], vein['y']),
                vein['radius'],
                class_id,
                -1
            )

        return mask

    def _create_fascia_mask(self, annotation: Dict) -> np.ndarray:
        """Create fascia detection mask"""
        h, w = annotation['image_size']
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw horizontal line for fascia
        y = annotation['fascia_y']
        thickness = 10
        cv2.line(mask, (0, y), (w, y), 1, thickness)

        return mask


class VeinDatasetBuilder:
    """Builder for creating train/val/test splits"""

    @staticmethod
    def from_sample_data(
        sample_data_root: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        frame_stride: int = 5,
        image_size: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders from Sample_Data folder structure

        Expected structure:
        Sample_Data/
        └── Set 1/
            ├── 0 - Raw videos/
            ├── 1 - Videos/
            ├── 2 - Annotated videos/
            └── 3 - Simple Annotated videos/
        """
        sample_data_root = Path(sample_data_root)

        # Collect video paths
        annotated_dir = sample_data_root / "Set 1" / "2 - Annotated videos"
        simple_annotated_dir = sample_data_root / "Set 1" / "3 - Simple Annotated videos"
        raw_dir = sample_data_root / "Set 1" / "1 - Videos"

        video_paths = []

        # Use fully annotated videos
        if annotated_dir.exists():
            video_paths.extend(list(annotated_dir.glob("*.mp4")))
            logger.info(f"Found {len(list(annotated_dir.glob('*.mp4')))} annotated videos")

        # Also use simple annotated videos
        if simple_annotated_dir.exists():
            video_paths.extend(list(simple_annotated_dir.glob("*.mp4")))
            logger.info(f"Found {len(list(simple_annotated_dir.glob('*.mp4')))} simple annotated videos")

        if not video_paths:
            logger.warning("No videos found in Sample_Data")
            # Fallback to raw videos
            if raw_dir.exists():
                video_paths.extend(list(raw_dir.glob("*.mp4")))
                logger.info(f"Falling back to {len(list(raw_dir.glob('*.mp4')))} raw videos")

        logger.info(f"Total videos found: {len(video_paths)}")

        # Train/val/test split
        train_paths, test_paths = train_test_split(
            video_paths, test_size=(val_ratio + test_ratio), random_state=42
        )
        val_paths, test_paths = train_test_split(
            test_paths,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42
        )

        logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

        # Create datasets
        train_dataset = UltrasoundVeinDataset(
            train_paths,
            frame_stride=frame_stride,
            image_size=image_size,
            augment=True
        )

        val_dataset = UltrasoundVeinDataset(
            val_paths,
            frame_stride=frame_stride,
            image_size=image_size,
            augment=False
        )

        test_dataset = UltrasoundVeinDataset(
            test_paths,
            frame_stride=frame_stride,
            image_size=image_size,
            augment=False
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    logging.basicConfig(level=logging.INFO)

    sample_data_root = Path("/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data")

    train_loader, val_loader, test_loader = VeinDatasetBuilder.from_sample_data(
        sample_data_root,
        batch_size=2,
        frame_stride=10
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a sample batch
    for batch in train_loader:
        print(f"\nSample batch shapes:")
        print(f"  image: {batch['image'].shape}")
        print(f"  vein_mask: {batch['vein_mask'].shape}")
        print(f"  classification_mask: {batch['classification_mask'].shape}")
        print(f"  fascia_mask: {batch['fascia_mask'].shape}")
        break
