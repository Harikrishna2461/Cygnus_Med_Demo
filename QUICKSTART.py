#!/usr/bin/env python3
"""
QUICK START - VEIN DETECTION SYSTEM
Complete setup and training on real BUSI dataset
"""

import subprocess
import sys
import os
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║          CYGNUS VEIN DETECTION - QUICK START GUIDE                      ║
║                   (Real Data Only - NO Synthetic)                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def run_command(cmd, description):
    """Run a command and show status."""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"✗ {description} - FAILED (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def check_file_exists(path, description):
    """Check if file exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False

def check_dataset():
    """Check if real dataset exists."""
    data_dir = Path('./backend/vision/segmentation/data/ultrasound_fascia')
    
    print(f"\n{'='*70}")
    print("Checking Dataset Status")
    print(f"{'='*70}")
    
    if (data_dir / 'images').exists():
        images = list((data_dir / 'images').glob('*'))
        masks = list((data_dir / 'masks').glob('*')) if (data_dir / 'masks').exists() else []
        
        print(f"✓ Dataset directory found: {data_dir}")
        print(f"  - Images: {len(images)} files")
        print(f"  - Masks: {len(masks)} files")
        
        if len(images) > 20:
            print(f"  ✓ Dataset size good for training!")
        elif len(images) >= 10:
            print(f"  ⚠ Small dataset (consider getting BUSI with 780 images)")
        else:
            print(f"  ✗ Dataset too small - download BUSI dataset")
            
        return len(images) > 0
    else:
        print(f"✗ Dataset directory not found: {data_dir}")
        return False

def menu():
    """Show menu options."""
    print(f"\n{'='*70}")
    print("WHAT WOULD YOU LIKE TO DO?")
    print(f"{'='*70}")
    print("""
1. ✓ Check System Status (verify all components)
2. 📥 Download BUSI Dataset Instructions (show how to download)
3. 📦 Prepare Dataset (organize BUSI for training)
4. 🏋️  Train Model (train UNet on real ultrasound data)
5. 🚀 Start Backend Server (run Flask app)
6. 🧪 Run Tests (verify all components)
7. 📊 Show Current Dataset Status
8. 📖 Show System Documentation
0. 🚪 Exit

    NOTE: Steps should be done in order: 1 → 2 → 3 → 4 → 5
""")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    while True:
        menu()
        
        try:
            choice = input("Enter your choice (0-8): ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        
        if choice == '1':
            # Check system status
            print("\n✓ Checking system components...")
            check_file_exists('./backend/app.py', 'Flask backend')
            check_file_exists('./backend/vision/segmentation/unet_fascia.py', 'UNet model')
            check_file_exists('./backend/vision/classification/vein_classifier.py', 'Vein classifier')
            check_file_exists('./frontend/src/App.js', 'React frontend')
            check_dataset()
            
        elif choice == '2':
            # Download instructions
            run_command('python3 BUSI_DOWNLOAD_GUIDE.py', 'Show BUSI Download Guide')
            
        elif choice == '3':
            # Prepare dataset
            busi_path = input("Enter path to extracted BUSI dataset: ").strip()
            if busi_path:
                cmd = f"python3 download_prepare_busi.py {busi_path}"
                run_command(cmd, f"Prepare BUSI Dataset from {busi_path}")
            else:
                print("✗ No path provided")
                
        elif choice == '4':
            # Train model
            if check_dataset():
                run_command('python3 setup_and_train.py', 'Train UNet Fascia Model on Real Data')
            else:
                print("\n✗ Dataset not found. Download BUSI first (option 2-3)")
                
        elif choice == '5':
            # Start backend
            print("\n" + "="*70)
            print("Starting Flask Backend Server...")
            print("="*70)
            print("Server will run on http://localhost:5000")
            print("Press Ctrl+C to stop")
            print("\nAPI Endpoints:")
            print("  POST /api/vision/analyze-integrated-veins - Image analysis")
            print("  POST /api/vision/analyze-integrated-video - Video analysis")
            print("  POST /api/vision/analyze-fascia - Fascia segmentation only")
            print("  GET  /api/vision/health - System health check")
            print("="*70 + "\n")
            
            os.chdir('./backend')
            run_command('python3 app.py', 'Flask API Server')
            
        elif choice == '6':
            # Run tests
            run_command('python3 test_vein_system.py', 'System Integration Tests')
            
        elif choice == '7':
            # Show dataset status
            check_dataset()
            
        elif choice == '8':
            # Show documentation
            if Path('SYSTEM_STATUS.md').exists():
                with open('SYSTEM_STATUS.md', 'r') as f:
                    print(f.read())
            else:
                print("✗ SYSTEM_STATUS.md not found")
                
        elif choice == '0':
            print("\nGoodbye!")
            sys.exit(0)
            
        else:
            print("✗ Invalid choice. Please enter 0-8")
