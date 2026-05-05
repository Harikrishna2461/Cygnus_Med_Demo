import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
