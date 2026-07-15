"""
Training script for UNet fascia segmentation model.
Trains on ultrasound dataset to detect fascia boundaries.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
import json

from vision.segmentation.unet_fascia import UNetFascia, FasciaDetector
from vision.segmentation.ultrasound_dataset import create_dataloaders, DatasetDownloader


class FasciaTrainer:
    """Trainer for UNet fascia segmentation model."""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss function: Binary Cross Entropy with Dice Loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.history = {'train': [], 'test': []}
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice coefficient loss."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = 2.0 * intersection / (union + smooth)
        return 1.0 - dice
    
    def combined_loss(self, pred, target):
        """Combined BCE + Dice loss."""
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # Loss
            loss = self.combined_loss(logits, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model(images)
                loss = self.combined_loss(logits, masks)
                
                # Dice score
                pred = torch.sigmoid(logits)
                intersection = (pred * masks).sum()
                dice = 2.0 * intersection / (pred.sum() + masks.sum() + 1e-5)
                
                total_loss += loss.item()
                total_dice += dice.item()
        
        avg_loss = total_loss / len(test_loader)
        avg_dice = total_dice / len(test_loader)
        
        return avg_loss, avg_dice
    
    def train(self, train_loader, test_loader, epochs=30, save_interval=5):
        """
        Train for multiple epochs.
        
        Args:
            train_loader: training dataloader
            test_loader: test dataloader
            epochs: number of epochs
            save_interval: save checkpoint every N epochs
        """
        best_test_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train'].append(train_loss)
            
            # Evaluate
            test_loss, test_dice = self.evaluate(test_loader)
            self.history['test'].append(test_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss:  {test_loss:.4f}")
            print(f"Test Dice:  {test_dice:.4f}")
            
            self.scheduler.step(test_loss)
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_checkpoint(f'unet_fascia_best.pth')
                print("✓ Saved best model")
            
            # Save periodic checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(f'unet_fascia_epoch_{epoch}.pth')
        
        print(f"\n✓ Training complete. Best test loss: {best_test_loss:.4f}")
        self.save_history()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
    
    def save_history(self):
        """Save training history."""
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main training pipeline."""
    
    # Configuration
    DATA_DIR = Path('./backend/vision/segmentation/data/ultrasound_fascia')
    CHECKPOINT_DIR = Path('./backend/vision/segmentation/checkpoints')
    MODEL_PATH = CHECKPOINT_DIR / 'unet_fascia_best.pth'
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*50)
    print("Preparing dataset...")
    print("="*50)
    
    if not DATA_DIR.exists() or not (DATA_DIR / 'images').exists():
        print(f"\n✗ FATAL: Real ultrasound dataset NOT FOUND")
        print(f"Expected: {DATA_DIR}/")
        print("          ├── images/")
        print("          └── masks/")
        print("\nTo prepare real data:")
        print("  1. Download BUSI: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
        print("  2. Run: python3 download_prepare_busi.py <path_to_dataset>")
        print("\n✗ NO SYNTHETIC DATA - USING REAL DATASETS ONLY")
        import sys
        sys.exit(1)
    else:
        print(f"✓ Dataset found at {DATA_DIR}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        str(DATA_DIR),
        batch_size=BATCH_SIZE,
        augment=True
    )
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    model = UNetFascia(in_channels=3, out_channels=1, num_filters=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = FasciaTrainer(
        model,
        device=device,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=str(CHECKPOINT_DIR)
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    trainer.train(
        train_loader,
        test_loader,
        epochs=EPOCHS,
        save_interval=5
    )
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"✓ Best model saved to: {MODEL_PATH}")
    print(f"✓ Checkpoints saved to: {CHECKPOINT_DIR}")
    
    # Test loading model
    print("\nTesting model loading...")
    detector = FasciaDetector(model_path=str(MODEL_PATH), device=device)
    print("✓ Model loaded successfully")
    
    return model_path


if __name__ == '__main__':
    main()
