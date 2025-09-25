#!/usr/bin/env python
"""Simple trace to check factory calls."""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Patch the factory function
from dendritic_modeling.networks.architectures import factory

original_get_architecture = factory.get_architecture
call_count = 0


def traced_get_architecture(*args, **kwargs):
    global call_count
    call_count += 1
    start = time.time()
    logging.info(
        f"get_architecture call #{call_count} - type: {args[0] if args else 'None'}"
    )
    result = original_get_architecture(*args, **kwargs)
    elapsed = time.time() - start
    logging.info(f"get_architecture call #{call_count} completed in {elapsed:.3f}s")
    return result


factory.get_architecture = traced_get_architecture

# Also patch prepare_ei_network_params
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    config_adapter,
)

original_prepare = config_adapter.prepare_ei_network_params
prepare_count = 0


def traced_prepare(*args, **kwargs):
    global prepare_count
    prepare_count += 1
    start = time.time()
    logging.info(f"prepare_ei_network_params call #{prepare_count}")
    result = original_prepare(*args, **kwargs)
    elapsed = time.time() - start
    logging.info(
        f"prepare_ei_network_params call #{prepare_count} completed in {elapsed:.3f}s"
    )
    return result


config_adapter.prepare_ei_network_params = traced_prepare

# Now just create a model
logging.info("Creating model...")
from dendritic_modeling.scripts.script_utils.config_utils import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment

config = load_config(".vscode/config_exp.yaml")
model, datasets, trainer_config = setup_environment(config)

logging.info("\nSUMMARY:")
logging.info(f"get_architecture was called {call_count} times")
logging.info(f"prepare_ei_network_params was called {prepare_count} times")
