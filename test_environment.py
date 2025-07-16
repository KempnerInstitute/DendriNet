#!/usr/bin/env python
"""Test script to verify environment setup"""

import sys
import os

print("=== Environment Test ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check if we can import PyTorch
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"\nERROR: Could not import PyTorch: {e}")

# Check distributed training
try:
    import torch.distributed as dist
    print(f"\nDistributed package available: True")
except ImportError:
    print(f"\nDistributed package available: False")

print("\n=== Test Complete ===")
