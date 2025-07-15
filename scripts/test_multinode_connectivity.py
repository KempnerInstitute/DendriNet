#!/usr/bin/env python
"""
Simple script to test multi-node connectivity with PyTorch distributed.
This helps diagnose networking issues before running full training.
"""

import os
import socket
import torch
import torch.distributed as dist
from datetime import timedelta

def get_ip_address():
    """Get the IP address of the current node."""
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except:
        # Fallback to getting IP from interface
        import subprocess
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        ip = result.stdout.strip().split()[0] if result.stdout else 'unknown'
    return ip

def main():
    # Get environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"[Rank {rank}] Starting connectivity test...")
    print(f"[Rank {rank}] Environment:")
    print(f"  - RANK: {rank}")
    print(f"  - LOCAL_RANK: {local_rank}")
    print(f"  - WORLD_SIZE: {world_size}")
    print(f"  - MASTER_ADDR: {master_addr}")
    print(f"  - MASTER_PORT: {master_port}")
    print(f"  - Hostname: {socket.gethostname()}")
    print(f"  - IP Address: {get_ip_address()}")
    
    # Try to initialize process group with longer timeout
    try:
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10)
        )
        print(f"[Rank {rank}] Process group initialized successfully!")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            print(f"[Rank {rank}] Using device: {device}")
        else:
            device = torch.device('cpu')
            print(f"[Rank {rank}] WARNING: No CUDA devices available, using CPU")
        
        # Test 1: All-reduce
        print(f"[Rank {rank}] Testing all-reduce...")
        tensor = torch.ones(1).to(device) * (rank + 1)
        print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")
        dist.all_reduce(tensor)
        expected = sum(range(1, world_size + 1))
        print(f"[Rank {rank}] After all-reduce: {tensor.item()} (expected: {expected})")
        
        # Test 2: Broadcast
        print(f"[Rank {rank}] Testing broadcast...")
        if rank == 0:
            tensor = torch.tensor([42.0]).to(device)
        else:
            tensor = torch.zeros(1).to(device)
        dist.broadcast(tensor, src=0)
        print(f"[Rank {rank}] After broadcast: {tensor.item()} (expected: 42.0)")
        
        # Test 3: Barrier
        print(f"[Rank {rank}] Testing barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Barrier completed")
        
        print(f"[Rank {rank}] All tests passed!")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"[Rank {rank}] Process group destroyed")

if __name__ == "__main__":
    main() 