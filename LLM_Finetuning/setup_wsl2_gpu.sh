#!/bin/bash
# WSL2 GPU Setup for CHIVA LoRA Training
# Run this ONCE in WSL2: bash setup_wsl2_gpu.sh

set -e

echo "=========================================="
echo "WSL2 GPU Setup for PyTorch + CUDA 12.9"
echo "=========================================="

# Update package manager
echo "[1/5] Updating apt packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv -qq

# Create virtual environment (optional but recommended)
echo "[2/5] Creating Python virtual environment..."
python3 -m venv ~/gpu_env
source ~/gpu_env/bin/activate

# Install PyTorch for CUDA 12.4 (compatible with CUDA 12.9)
echo "[3/5] Installing PyTorch with CUDA 12.4 support..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
echo "[4/5] Installing training dependencies..."
pip install -q transformers peft datasets tensorboard

# Verify CUDA
echo "[5/5] Verifying GPU access..."
python3 << 'VERIFY'
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
VERIFY

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next: Open Jupyter and run the notebook cells"
echo "  jupyter notebook"
echo ""
echo "Activate environment in future shells:"
echo "  source ~/gpu_env/bin/activate"
echo ""
