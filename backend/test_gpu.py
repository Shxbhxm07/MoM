#!/usr/bin/env python3
"""Test GPU detection before starting main server"""

import torch
import platform

print("=" * 60)
print("GPU DETECTION TEST")
print("=" * 60)

# System info
print(f"System: {platform.system()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

# GPU detection
if torch.cuda.is_available():
    print(f"\n✅ CUDA is AVAILABLE!")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Test allocation
    try:
        x = torch.randn(100, 100).cuda()
        print(f"   ✅ Test tensor allocated on GPU successfully")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ GPU allocation failed: {e}")
else:
    print(f"\n❌ CUDA is NOT available")
    print(f"   Reason: Check if PyTorch was installed with CUDA support")

print("\n" + "=" * 60)
print("Now the main app will determine device...")
print("=" * 60 + "\n")

# Import main to see what device it picks
import main
