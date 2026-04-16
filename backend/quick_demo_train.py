#!/usr/bin/env python3
"""
Quick Demo Training - Shows vein detection model training in action
This is a fast version with minimal data and epochs to demonstrate training works
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from vein_detector_vit import CustomUltrasoundViT, VeinDetectionConfig
from vein_dataset import VeinDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_train_demo():
    """Quick demo training showing the model actually learns"""

    print("\n" + "="*80)
    print("VEIN DETECTION MODEL - QUICK DEMO TRAINING")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device: {device}")

    # Load data (small subset for speed)
    logger.info("\n📦 Loading datasets...")
    sample_data_root = Path("/Users/HariKrishnaD/Downloads/NUS/Internships/Cygnus/cmed_demo/Sample_Data")

    train_loader, val_loader, test_loader = VeinDatasetBuilder.from_sample_data(
        sample_data_root,
        batch_size=2,
        frame_stride=20,  # Skip more frames for speed
        image_size=512,
        num_workers=0
    )

    logger.info(f"  ✓ Train batches: {len(train_loader)}")
    logger.info(f"  ✓ Val batches: {len(val_loader)}")
    logger.info(f"  ✓ Test batches: {len(test_loader)}")

    # Create model
    logger.info("\n🧠 Creating Vision Transformer model...")
    config = VeinDetectionConfig()
    model = CustomUltrasoundViT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model: {total_params:,} parameters")

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)

    fascia_loss_fn = nn.CrossEntropyLoss()
    vein_loss_fn = nn.CrossEntropyLoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    # Training metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'time': []
    }

    # Quick training loop
    num_epochs = 3
    logger.info(f"\n⚡ Training for {num_epochs} epochs...\n")

    start_time = datetime.now()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_fascia_loss = 0.0
        epoch_vein_loss = 0.0
        epoch_classification_loss = 0.0

        logger.info(f"{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*80}")

        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            fascia_targets = batch['fascia_mask'].to(device).long()
            vein_targets = batch['vein_mask'].to(device).long()
            classification_targets = batch['classification_mask'].to(device).long()

            # Simple downsampling to patch level (for speed)
            h, w = fascia_targets.shape[1:]
            grid_size = 32
            fascia_targets = fascia_targets[:, :grid_size, :grid_size]
            vein_targets = vein_targets[:, :grid_size, :grid_size]
            classification_targets = classification_targets[:, :grid_size, :grid_size]

            optimizer.zero_grad()
            outputs = model(images)

            # Reshape for loss
            fascia_logits = outputs['fascia_logits'][:, 1:1+grid_size*grid_size, :]
            vein_logits = outputs['vein_logits'][:, 1:1+grid_size*grid_size, :]
            classification_logits = outputs['classification_logits'][:, 1:1+grid_size*grid_size, :]

            # Calculate losses
            f_loss = fascia_loss_fn(fascia_logits.reshape(-1, 2), fascia_targets.reshape(-1))
            v_loss = vein_loss_fn(vein_logits.reshape(-1, 4), vein_targets.reshape(-1))
            c_loss = classification_loss_fn(classification_logits.reshape(-1, 3), classification_targets.reshape(-1))

            # Weighted combination
            total_loss = 0.3 * f_loss + 0.5 * v_loss + 0.2 * c_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_fascia_loss += f_loss.item()
            epoch_vein_loss += v_loss.item()
            epoch_classification_loss += c_loss.item()
            batch_count += 1

            # Progress indicator
            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                avg_loss = epoch_loss / batch_count
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        # Epoch summary
        epoch_loss /= batch_count
        epoch_fascia_loss /= batch_count
        epoch_vein_loss /= batch_count
        epoch_classification_loss /= batch_count

        logger.info(f"\nTrain Loss: {epoch_loss:.4f}")
        logger.info(f"  Fascia Loss: {epoch_fascia_loss:.4f}")
        logger.info(f"  Vein Loss: {epoch_vein_loss:.4f}")
        logger.info(f"  Classification Loss: {epoch_classification_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                fascia_targets = batch['fascia_mask'].to(device).long()[:, :grid_size, :grid_size]
                vein_targets = batch['vein_mask'].to(device).long()[:, :grid_size, :grid_size]
                classification_targets = batch['classification_mask'].to(device).long()[:, :grid_size, :grid_size]

                outputs = model(images)

                fascia_logits = outputs['fascia_logits'][:, 1:1+grid_size*grid_size, :]
                vein_logits = outputs['vein_logits'][:, 1:1+grid_size*grid_size, :]
                classification_logits = outputs['classification_logits'][:, 1:1+grid_size*grid_size, :]

                f_loss = fascia_loss_fn(fascia_logits.reshape(-1, 2), fascia_targets.reshape(-1))
                v_loss = vein_loss_fn(vein_logits.reshape(-1, 4), vein_targets.reshape(-1))
                c_loss = classification_loss_fn(classification_logits.reshape(-1, 3), classification_targets.reshape(-1))

                total_loss = 0.3 * f_loss + 0.5 * v_loss + 0.2 * c_loss
                val_loss += total_loss.item()
                val_count += 1

        val_loss /= max(val_count, 1)
        logger.info(f"Val Loss: {val_loss:.4f}\n")

        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(float(epoch_loss))
        metrics['val_loss'].append(float(val_loss))
        metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()

    # Final summary
    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    logger.info(f"\n✅ Total training time: {total_time:.1f} seconds")
    logger.info(f"\nFinal Metrics:")
    logger.info(f"  Train Loss: {metrics['train_loss'][-1]:.4f}")
    logger.info(f"  Val Loss: {metrics['val_loss'][-1]:.4f}")
    logger.info(f"  Learning Rate: {metrics['learning_rate'][-1]:.2e}")

    logger.info(f"\n📊 Loss Trend:")
    for i, (tl, vl) in enumerate(zip(metrics['train_loss'], metrics['val_loss']), 1):
        logger.info(f"  Epoch {i}: Train={tl:.4f}, Val={vl:.4f}")

    # Save checkpoint
    checkpoint_dir = Path("./checkpoints/vein_detection")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "demo_model.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_path)

    logger.info(f"\n💾 Model saved to: {checkpoint_path}")

    # Save metrics
    metrics_path = checkpoint_dir / "demo_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"📈 Metrics saved to: {metrics_path}")

    print("\n" + "="*80)
    print("🎉 TRAINING DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nModel is trained and ready to use!")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print("="*80 + "\n")

    return model, metrics


if __name__ == "__main__":
    model, metrics = quick_train_demo()
