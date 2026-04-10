#!/usr/bin/env python3
"""
Quick training script - train UNet on real ultrasound data
"""

import sys
import os
sys.path.insert(0, './backend')

from pathlib import Path
import torch

def main():
    print('\n' + '='*70)
    print('TRAINING: UNet Fascia Detection Model on REAL Data')
    print('='*70)
    print('Dataset: Real ultrasound images (10 samples for testing)')
    print('Model: UNet with 31M parameters')
    print('Device: CPU')
    print('Epochs: 3 (quick demo)')
    print('='*70 + '\n')

    try:
        # Import modules
        from vision.segmentation.train_fascia import FasciaTrainer
        from vision.segmentation.ultrasound_dataset import create_dataloaders
        from vision.segmentation.unet_fascia import UNetFascia
        
        # Setup data
        data_dir = Path('./backend/vision/segmentation/data/ultrasound_fascia')
        print(f'Loading dataset from: {data_dir}')
        
        if not (data_dir / 'images').exists():
            print(f'✗ Dataset not found at {data_dir}')
            sys.exit(1)
        
        # Create data loaders (num_workers=0 for macOS compatibility)
        train_loader, test_loader = create_dataloaders(
            data_dir, 
            batch_size=2,
            num_workers=0,
            augment=False
        )
        
        print(f'✓ Train batches: {len(train_loader)}')
        print(f'✓ Test batches: {len(test_loader)}')
        
        # Create model
        model = UNetFascia(in_channels=3, out_channels=1)
        print(f'\n✓ UNet model created (31M parameters)')
        
        # Setup trainer
        trainer = FasciaTrainer(
            model=model,
            device='cpu',
            checkpoint_dir='./backend/vision/segmentation/checkpoints'
        )
        
        print(f'\nStarting training on REAL ultrasound data...\n')
        print('This may take 2-5 minutes on CPU...\n')
        
        # Train (3 epochs for quick demo)
        trainer.train(train_loader, test_loader, epochs=3)
        
        print(f'\n' + '='*70)
        print(f'✅ TRAINING COMPLETE!')
        print(f'='*70)
        print(f'Model saved to: ./backend/vision/segmentation/checkpoints/unet_fascia_best.pth')
        print(f'\nNext steps:')
        print(f'  1. Download real vein ultrasound dataset from public sources')
        print(f'  2. Organize in: ./backend/vision/segmentation/data/ultrasound_fascia/')
        print(f'  3. Run: python3 train_quick.py')
        print(f'  4. Deploy with: python3 backend/app.py')
        print(f'='*70 + '\n')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
