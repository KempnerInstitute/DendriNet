# FSDP Troubleshooting Guide

This guide addresses common issues with FSDP (Fully Sharded Data Parallel) training in the dendritic modeling framework.

## Common Error: NCCL Communication Timeout

### Symptoms
```
[rank3]:[E718 14:54:42.627298329 ProcessGroupNCCL.cpp:632] [Rank 3] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=7449, OpType=_ALLGATHER_BASE, NumelIn=260739, NumelOut=1042956, Timeout(ms)=600000) ran for 600000 milliseconds before timing out.
```

### Root Causes
1. **Network configuration issues** - IPv6/IPv4 conflicts
2. **Insufficient network bandwidth** - Large models overwhelming network
3. **Memory pressure** - Too large batches or models
4. **Hardware issues** - Faulty GPUs or network cards

## Solutions

### 1. Environment Variables (Already Applied)
The script now automatically sets these NCCL environment variables:
```bash
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800              # 30 minutes timeout
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0,eth0    # Try InfiniBand first
export NCCL_IB_DISABLE=0              # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2           # Enable GPU Direct RDMA
```

### 2. Reduce Model/Batch Size
If the timeout persists, try reducing the computational load:

```yaml
# In your config file
train:
  main:
    trainer:
      batch_size: 256    # Reduce from 500
      
model:
  core_network:
    parameters:
      excitatory_layer_sizes: [512, 256]  # Reduce from [1024, 512, 256]
      inhibitory_layer_sizes: [128, 64]   # Reduce from [256, 128, 64]
```

### 3. Enable CPU Offloading
For very large models, enable CPU offloading:

```yaml
# In your config file - add to trainer section
train:
  main:
    trainer:
      # FSDP-specific parameters
      use_mixed_precision: true
      cpu_offload: true        # Enable CPU offloading
      backward_prefetch: true
      limit_all_gathers: true
      forward_prefetch: true
```

### 4. Alternative NCCL Settings
If InfiniBand is causing issues, try Ethernet only:

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

### 5. Reduce Number of GPUs
As a last resort, use fewer GPUs:
```bash
# Instead of 4 GPUs
torchrun --nproc_per_node=4 train_large_model_fsdp.py config.yaml

# Try 2 GPUs
torchrun --nproc_per_node=2 train_large_model_fsdp.py config.yaml
```

## Network Configuration Issues

### IPv6 Socket Errors
```
[W718 14:44:00.805079375 socket.cpp:755] [c10d] The client socket cannot be initialized to connect to [localhost]:45213 (errno: 97 - Address family not supported by protocol).
```

**Solution**: Force IPv4 usage:
```bash
export NCCL_SOCKET_FAMILY=AF_INET
```

### Hostname Resolution Issues
If you see errors with hostnames like `holygpu8a10302.rc.fas.harvard.edu`:

**Solution**: Add to your SLURM script or environment:
```bash
export MASTER_ADDR=$(hostname -I | awk '{print $1}')  # Use IP instead of hostname
export NCCL_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}')  # Auto-detect interface
```

## Hardware-Specific Solutions

### For Harvard FAS-RC Systems
```bash
# Add to your SLURM script
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Set specific network interface
export NCCL_SOCKET_IFNAME=ib0  # For InfiniBand nodes
# OR
export NCCL_SOCKET_IFNAME=eth0  # For Ethernet nodes
```

### For Systems with Mixed Hardware
```bash
# Disable GPU Direct RDMA if causing issues
export NCCL_NET_GDR_LEVEL=0
```

## Debugging Steps

### 1. Test Communication
Before running training, test NCCL:
```bash
# Install nccl-tests (if available)
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
mpirun -np 4 ./build/all_reduce_perf -b 8 -e 256M -f 2
```

### 2. Monitor GPU Memory
```bash
# In another terminal during training
watch -n 1 nvidia-smi
```

### 3. Check Network Interface
```bash
# Check available network interfaces
ip link show
# Check InfiniBand status (if available)
ibstatus
```

## Configuration Examples

### Robust Configuration for Unstable Networks
```yaml
train:
  main:
    trainer:
      batch_size: 128           # Conservative batch size
      use_mixed_precision: true
      cpu_offload: false        # Start without CPU offload
      backward_prefetch: true
      limit_all_gathers: true
      forward_prefetch: true
      save_every_n_epochs: 5   # Save more frequently in case of crashes
```

### Large Model Configuration
```yaml
train:
  main:
    trainer:
      batch_size: 64            # Smaller batch for large models
      use_mixed_precision: true
      cpu_offload: true         # Enable for very large models
      backward_prefetch: true
      limit_all_gathers: true
      forward_prefetch: true
      save_every_n_epochs: 1   # Save every epoch for safety
```

## Recovery from Timeout

If training times out, you can resume from a checkpoint:

1. **Find the latest checkpoint**:
   ```bash
   ls -la results/your_run_name/checkpoints/
   ```

2. **Resume training** (feature to be implemented):
   ```bash
   torchrun --nproc_per_node=4 train_large_model_fsdp.py config.yaml --resume_from_checkpoint results/your_run_name/checkpoints/checkpoint_epoch_X.pt
   ```

## Performance Monitoring

Monitor these metrics during training:
- GPU utilization (`nvidia-smi`)
- Network utilization (`iftop`, `netstat`)
- Memory usage (both GPU and CPU)
- NCCL operation timing (from NCCL_DEBUG=INFO output)

## When to Use Single GPU Training

Consider using `train_experiments.py` instead if:
- Model fits on single GPU (< 24GB for RTX 4090, < 80GB for A100)
- Network issues persist despite troubleshooting
- Training is for experimentation rather than production

## Additional Resources

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [PyTorch Distributed Troubleshooting](https://pytorch.org/docs/stable/distributed.html#troubleshooting) 