"""
Training pipeline for vein detection Vision Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
import logging
import json
from datetime import datetime
import numpy as np

from vein_detector_vit import CustomUltrasoundViT, VeinDetectionConfig, VeinDetectionPostProcessor
from vein_dataset import VeinDatasetBuilder

logger = logging.getLogger(__name__)


class VeinDetectionTrainer:
    """Trainer for vein detection model"""

    def __init__(
        self,
        model: CustomUltrasoundViT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # Loss functions
        self.fascia_criterion = nn.CrossEntropyLoss()
        self.vein_criterion = nn.CrossEntropyLoss()
        self.classification_criterion = nn.CrossEntropyLoss()

        # Metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_fascia_acc': [],
            'val_fascia_acc': [],
            'train_vein_iou': [],
            'val_vein_iou': [],
        }

        self.best_val_loss = float('inf')
        self.post_processor = VeinDetectionPostProcessor()

    def _patch_targets(self, mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """Convert pixel-level targets to patch-level targets"""
        batch_size, h, w = mask.shape
        grid_size = h // patch_size

        # Average pool to patch level
        mask_reshaped = mask.view(batch_size, grid_size, patch_size, grid_size, patch_size)
        mask_reshaped = mask_reshaped.permute(0, 1, 3, 2, 4).contiguous()
        mask_reshaped = mask_reshaped.view(batch_size, grid_size * grid_size, -1)

        # Use majority voting for classification
        patch_targets = torch.mode(mask_reshaped, dim=2)[0]

        # Add CLS token position (dummy label)
        cls_token = torch.zeros(batch_size, 1, dtype=patch_targets.dtype, device=patch_targets.device)
        patch_targets = torch.cat([cls_token, patch_targets], dim=1)

        return patch_targets

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {
            'fascia_loss': 0.0,
            'vein_loss': 0.0,
            'classification_loss': 0.0,
            'fascia_acc': 0.0,
            'vein_iou': 0.0,
        }

        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            fascia_targets = self._patch_targets(batch['fascia_mask'].to(self.device))
            vein_targets = self._patch_targets(batch['vein_mask'].to(self.device))
            classification_targets = self._patch_targets(batch['classification_mask'].to(self.device))

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Calculate losses
            fascia_loss = self.fascia_criterion(
                outputs['fascia_logits'].view(-1, 2),
                fascia_targets.view(-1)
            )

            vein_loss = self.vein_criterion(
                outputs['vein_logits'].view(-1, 4),  # background + 3 classes
                vein_targets.view(-1)
            )

            classification_loss = self.classification_criterion(
                outputs['classification_logits'].view(-1, 3),  # N1, N2, N3
                classification_targets.view(-1)
            )

            # Weighted combination
            total_loss = 0.3 * fascia_loss + 0.5 * vein_loss + 0.2 * classification_loss

            # Backward pass
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += total_loss.item()
            epoch_metrics['fascia_loss'] += fascia_loss.item()
            epoch_metrics['vein_loss'] += vein_loss.item()
            epoch_metrics['classification_loss'] += classification_loss.item()

            # Calculate accuracies
            fascia_pred = torch.argmax(outputs['fascia_logits'], dim=-1)
            fascia_acc = (fascia_pred == fascia_targets).float().mean().item()
            epoch_metrics['fascia_acc'] += fascia_acc

            vein_pred = torch.argmax(outputs['vein_logits'], dim=-1)
            vein_iou = self._calculate_iou(vein_pred, vein_targets)
            epoch_metrics['vein_iou'] += vein_iou

            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} - "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Fascia Acc: {fascia_acc:.4f}, "
                    f"Vein IoU: {vein_iou:.4f}"
                )

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        epoch_metrics['loss'] = epoch_loss / num_batches
        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {
            'fascia_loss': 0.0,
            'vein_loss': 0.0,
            'classification_loss': 0.0,
            'fascia_acc': 0.0,
            'vein_iou': 0.0,
        }

        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            fascia_targets = self._patch_targets(batch['fascia_mask'].to(self.device))
            vein_targets = self._patch_targets(batch['vein_mask'].to(self.device))
            classification_targets = self._patch_targets(batch['classification_mask'].to(self.device))

            # Forward pass
            outputs = self.model(images)

            # Calculate losses
            fascia_loss = self.fascia_criterion(
                outputs['fascia_logits'].view(-1, 2),
                fascia_targets.view(-1)
            )

            vein_loss = self.vein_criterion(
                outputs['vein_logits'].view(-1, 4),
                vein_targets.view(-1)
            )

            classification_loss = self.classification_criterion(
                outputs['classification_logits'].view(-1, 3),
                classification_targets.view(-1)
            )

            total_loss = 0.3 * fascia_loss + 0.5 * vein_loss + 0.2 * classification_loss

            epoch_loss += total_loss.item()
            epoch_metrics['fascia_loss'] += fascia_loss.item()
            epoch_metrics['vein_loss'] += vein_loss.item()
            epoch_metrics['classification_loss'] += classification_loss.item()

            # Calculate accuracies
            fascia_pred = torch.argmax(outputs['fascia_logits'], dim=-1)
            fascia_acc = (fascia_pred == fascia_targets).float().mean().item()
            epoch_metrics['fascia_acc'] += fascia_acc

            vein_pred = torch.argmax(outputs['vein_logits'], dim=-1)
            vein_iou = self._calculate_iou(vein_pred, vein_targets)
            epoch_metrics['vein_iou'] += vein_iou

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        epoch_metrics['loss'] = epoch_loss / num_batches
        return epoch_metrics

    @staticmethod
    def _calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Intersection over Union"""
        pred_binary = (pred > 0).float()
        target_binary = (target > 0).float()

        intersection = (pred_binary * target_binary).sum().item()
        union = (pred_binary + target_binary - pred_binary * target_binary).sum().item()

        if union == 0:
            return 0.0

        return intersection / union

    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Checkpoint loaded from {path}")

    def fit(
        self,
        num_epochs: int = 50,
        checkpoint_dir: Path = None,
        save_interval: int = 5,
    ):
        """Train model for specified number of epochs"""
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints/vein_detection")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Fascia Loss: {train_metrics['fascia_loss']:.4f}, Acc: {train_metrics['fascia_acc']:.4f}")
            logger.info(f"  Vein Loss: {train_metrics['vein_loss']:.4f}, IoU: {train_metrics['vein_iou']:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Fascia Loss: {val_metrics['fascia_loss']:.4f}, Acc: {val_metrics['fascia_acc']:.4f}")
            logger.info(f"  Vein Loss: {val_metrics['vein_loss']:.4f}, IoU: {val_metrics['vein_iou']:.4f}")

            # Store metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_fascia_acc'].append(train_metrics['fascia_acc'])
            self.metrics['val_fascia_acc'].append(val_metrics['fascia_acc'])
            self.metrics['train_vein_iou'].append(train_metrics['vein_iou'])
            self.metrics['val_vein_iou'].append(val_metrics['vein_iou'])

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch + 1)

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                best_path = checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path, epoch + 1)
                logger.info(f"Best model saved (val_loss: {self.best_val_loss:.4f})")

        logger.info("\nTraining completed!")
        return self.metrics


def train_vein_detector(
    sample_data_root: Path = None,
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    checkpoint_dir: Path = None,
):
    """Main training function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if sample_data_root is None:
        sample_data_root = Path("/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data")

    if checkpoint_dir is None:
        checkpoint_dir = Path("./checkpoints/vein_detection")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = VeinDatasetBuilder.from_sample_data(
        sample_data_root,
        batch_size=batch_size,
        frame_stride=5,
        image_size=512,
    )

    # Create model
    config = VeinDetectionConfig()
    model = CustomUltrasoundViT(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = VeinDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
    )

    # Train
    metrics = trainer.fit(
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        save_interval=5,
    )

    # Save final metrics
    metrics_path = checkpoint_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Train
    train_vein_detector(
        batch_size=4,
        num_epochs=10,
        learning_rate=1e-4,
    )
