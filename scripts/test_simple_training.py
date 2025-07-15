#!/usr/bin/env python
"""
Simple test script to verify training with performance plotting only.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dendritic_modeling.scripts.train_experiments import main

if __name__ == "__main__":
    # Run with the simple test configuration
    config_path = "configs/ei_fsdp_test_simple.yaml"
    
    # Override sys.argv to pass the config
    sys.argv = ["train_experiments.py", config_path]
    
    print(f"Running training with config: {config_path}")
    print("Only performance analysis is enabled")
    
    try:
        main()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc() 