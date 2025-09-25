#!/usr/bin/env python
"""Profile per-epoch timing to see if initialization happens repeatedly."""

import time

from dendritic_modeling.networks.architectures import factory
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    config_adapter,
)

# Track factory calls
factory_calls = []
prepare_calls = []

# Patch factory.get_architecture
original_get_architecture = factory.get_architecture


def traced_get_architecture(*args, **kwargs):
    call_time = time.time()
    factory_calls.append(("get_architecture", call_time, args[0] if args else None))
    return original_get_architecture(*args, **kwargs)


factory.get_architecture = traced_get_architecture

# Patch config_adapter.prepare_ei_network_params
original_prepare = config_adapter.prepare_ei_network_params


def traced_prepare(*args, **kwargs):
    call_time = time.time()
    prepare_calls.append(("prepare_ei_network_params", call_time))
    return original_prepare(*args, **kwargs)


config_adapter.prepare_ei_network_params = traced_prepare

# Patch OmegaConf.to_container to track config conversions
from omegaconf import OmegaConf

omega_calls = []
original_to_container = OmegaConf.to_container


def traced_to_container(*args, **kwargs):
    call_time = time.time()
    omega_calls.append(("OmegaConf.to_container", call_time))
    return original_to_container(*args, **kwargs)


OmegaConf.to_container = traced_to_container

print("Starting training with tracing enabled...")
print("=" * 60)

import sys

# Import and run training
from dendritic_modeling.scripts.training.train_experiments import main

sys.argv = [
    "train_experiments.py",
    ".vscode/config_exp.yaml",
    "training.epochs=3",  # Just 3 epochs
    "experiment.enable_hooks=true",
    "training.suppress_prints=false",
]

# Record epoch times
epoch_start_times = []
original_print = print


def patched_print(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    if "epoch=" in msg and "/100" in msg:
        epoch_start_times.append(time.time())
    return original_print(*args, **kwargs)


import builtins

builtins.print = patched_print

try:
    start_time = time.time()
    main(".vscode/config_exp.yaml")
    total_time = time.time() - start_time
except Exception as e:
    print(f"\nError during training: {e}")
    total_time = time.time() - start_time

# Restore original print
builtins.print = original_print

print("\n" + "=" * 60)
print("ANALYSIS RESULTS")
print("=" * 60)

print(f"\nTotal training time: {total_time:.2f}s")
print(f"\nFactory calls: {len(factory_calls)}")
for i, (name, t, type_arg) in enumerate(factory_calls):
    print(f"  Call {i+1}: {name}('{type_arg}') at t={t-start_time:.2f}s")

print(f"\nConfig adapter calls: {len(prepare_calls)}")
for i, (name, t) in enumerate(prepare_calls):
    print(f"  Call {i+1}: {name} at t={t-start_time:.2f}s")

print(f"\nOmegaConf.to_container calls: {len(omega_calls)}")
if len(omega_calls) > 10:
    print("  First 5 calls:")
    for i in range(5):
        _, t = omega_calls[i]
        print(f"    Call {i+1} at t={t-start_time:.2f}s")
    print("  ...")
    print("  Last 5 calls:")
    for i in range(len(omega_calls) - 5, len(omega_calls)):
        _, t = omega_calls[i]
        print(f"    Call {i+1} at t={t-start_time:.2f}s")
else:
    for i, (_, t) in enumerate(omega_calls):
        print(f"  Call {i+1} at t={t-start_time:.2f}s")

# Analyze epoch timing
if len(epoch_start_times) > 1:
    print("\nEpoch timing:")
    for i in range(1, len(epoch_start_times)):
        epoch_time = epoch_start_times[i] - epoch_start_times[i - 1]
        print(f"  Epoch {i}: {epoch_time:.2f}s")
    avg_epoch_time = (epoch_start_times[-1] - epoch_start_times[0]) / (
        len(epoch_start_times) - 1
    )
    print(f"  Average: {avg_epoch_time:.2f}s/epoch")

# Check if calls happen during training
if factory_calls and epoch_start_times:
    print("\nChecking if factory is called during training:")
    training_start = epoch_start_times[0] if epoch_start_times else start_time
    for name, t, type_arg in factory_calls:
        if t > training_start:
            print(f"  WARNING: {name} called DURING training at t={t-start_time:.2f}s")
