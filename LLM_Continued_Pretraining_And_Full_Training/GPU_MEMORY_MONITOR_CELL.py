# ============ COPY THIS ENTIRE CELL INTO JUPYTER ============
# Run this BEFORE training starts to monitor GPU memory in real-time

import torch
import psutil
from datetime import datetime

print("\n" + "="*80)
print("GPU MEMORY MONITOR")
print("="*80)

# Reset and clear
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

def show_memory(label=""):
    """Show current GPU memory state."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    free = total - reserved

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {label}")
    print(f"  Total GPU:     {total:.2f} GB")
    print(f"  Allocated:     {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved:      {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"  Free:          {free:.2f} GB ({free/total*100:.1f}%)")
    print(f"  Peak so far:   {peak:.2f} GB")
    print(f"  Fragmentation: {(reserved - allocated):.2f} GB")

    if allocated > total:
        print(f"\n  🔴 DANGER: Allocated ({allocated:.2f}GB) > Total ({total:.2f}GB)")
    elif free < 5:
        print(f"\n  ⚠️  WARNING: Less than 5GB free")
    else:
        print(f"\n  ✅ OK: {free:.2f}GB free")

# Show initial state
show_memory("BEFORE TRAINING")

# ============ NOW RUN YOUR TRAINING CELL AFTER THIS ============
# Then run the cell below multiple times to monitor during training:

print("\n" + "="*80)
print("Run this code in NEW CELLS during training to monitor:")
print("="*80)
print("""
# Copy-paste this into a NEW CELL and run it DURING training:
import torch
from datetime import datetime

total = torch.cuda.get_device_properties(0).total_memory / 1e9
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved = torch.cuda.memory_reserved(0) / 1e9
peak = torch.cuda.max_memory_allocated(0) / 1e9
free = total - reserved

print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU Memory:")
print(f"  Allocated: {allocated:.2f} / {total:.2f} GB ({allocated/total*100:.1f}%)")
print(f"  Reserved:  {reserved:.2f} GB")
print(f"  Free:      {free:.2f} GB")
print(f"  Peak:      {peak:.2f} GB")

if allocated > total:
    print(f"🔴 CRITICAL: Over capacity by {allocated - total:.2f}GB!")
elif free < 1:
    print("🔴 CRITICAL: Less than 1GB free!")
elif free < 5:
    print("⚠️  WARNING: Less than 5GB free!")
else:
    print(f"✅ OK")
""")
