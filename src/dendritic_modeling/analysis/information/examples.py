"""Example usage of the integrated information analysis framework.

This module demonstrates how to use the new unified information analysis
system with different estimation methods and configurations.
"""

import numpy as np
from typing import Dict, Any

# Import the integrated framework
from .integrated_analyzer import IntegratedInformationAnalyzer
from .config import (
    ConfigBuilder,
    InformationAnalysisPreset,
    load_config_from_dict,
    save_config_to_dict
)
from .estimators import EstimationMethod


def example_basic_usage():
    """Basic usage example with default configuration."""
    print("=== Basic Usage Example ===\n")
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    
    # Simulate excitatory and inhibitory inputs
    E = np.random.randn(n_samples, n_features)
    I = np.random.randn(n_samples, n_features)
    
    # Simulate branch output with some dependency on E and I
    Vout = 0.5 * E + 0.3 * I + 0.2 * np.random.randn(n_samples, n_features)
    
    # Class labels (discrete)
    C = np.random.randint(0, 3, n_samples)
    
    # Optional: soma activation
    Vinf = np.tanh(np.mean(Vout, axis=1))
    
    # Create analyzer with default configuration
    config = InformationAnalysisPreset.full_analysis(method="kraskov")
    analyzer = IntegratedInformationAnalyzer(config)
    
    # Run analysis
    results = analyzer.analyze(E, I, Vout, C, Vinf)
    
    # Print summary
    print(analyzer.get_summary(results))
    
    return results


def example_method_comparison():
    """Compare different estimation methods on the same data."""
    print("\n=== Method Comparison Example ===\n")
    
    # Generate data
    n_samples = 500
    E = np.random.randn(n_samples)
    I = np.random.randn(n_samples)
    Vout = 0.7 * E - 0.3 * I + 0.1 * np.random.randn(n_samples)
    C = (Vout > 0).astype(int)  # Binary classification based on Vout
    
    methods = ["kraskov", "sklearn", "copula"]
    results_by_method = {}
    
    for method in methods:
        print(f"\nAnalyzing with {method} method...")
        
        # Create configuration for this method
        config = ConfigBuilder() \
            .with_method(method) \
            .with_metrics(basic=True, pairwise=True, conditional=False, pid=False) \
            .build()
        
        # Create analyzer and run
        analyzer = IntegratedInformationAnalyzer(config)
        results = analyzer.analyze(E, I, Vout, C)
        
        results_by_method[method] = results
        
        # Print key metrics
        if "basic_mi" in results:
            print(f"  I(E;C) = {results['basic_mi']['I(E;C)']:.4f}")
            print(f"  I(I;C) = {results['basic_mi']['I(I;C)']:.4f}")
            print(f"  I(Vout;C) = {results['basic_mi']['I(Vout;C)']:.4f}")
    
    return results_by_method


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===\n")
    
    # Build custom configuration using the builder
    config = ConfigBuilder() \
        .with_method("kraskov") \
        .with_metrics(basic=True, pairwise=False, conditional=True, pid=True) \
        .with_kraskov_params(n_neighbors=5) \
        .with_pid_params(binarize=True, binarize_method="median") \
        .with_options(include_vinf=False, normalize_mi=True, verbose=True) \
        .build()
    
    # Or load from dictionary (e.g., from a config file)
    config_dict = {
        "estimator_config": {
            "method": "sklearn",
            "n_bins": 30,
        },
        "compute_basic_mi": True,
        "compute_pairwise_mi": False,
        "compute_conditional_mi": True,
        "compute_pid": False,
        "normalize_mi": True,
    }
    
    config_from_dict = load_config_from_dict(config_dict)
    
    # Save configuration for later use
    saved_config = save_config_to_dict(config)
    print(f"Saved configuration: {saved_config}")
    
    return config, config_from_dict


def example_conditional_mi_analysis():
    """Example focusing on conditional mutual information."""
    print("\n=== Conditional MI Analysis Example ===\n")
    
    # Generate data with specific dependencies
    n_samples = 800
    C = np.random.randint(0, 2, n_samples)  # Binary class
    
    # E and I are conditionally independent given C
    E = np.where(C == 0, 
                 np.random.normal(0, 1, n_samples),
                 np.random.normal(2, 1, n_samples))
    
    I = np.where(C == 0,
                 np.random.normal(0, 1, n_samples),
                 np.random.normal(-2, 1, n_samples))
    
    # Vout depends on both E and I
    Vout = E + I + 0.1 * np.random.randn(n_samples)
    
    # Use conditional analysis preset
    config = InformationAnalysisPreset.conditional_analysis(method="kraskov")
    analyzer = IntegratedInformationAnalyzer(config)
    
    results = analyzer.analyze(E, I, Vout, C)
    
    print("Conditional MI Results:")
    if "conditional_mi" in results:
        for metric, value in results["conditional_mi"].items():
            print(f"  {metric} = {value:.4f}")
    
    return results


def example_pid_analysis():
    """Example of Partial Information Decomposition."""
    print("\n=== PID Analysis Example ===\n")
    
    # Generate data with synergistic/redundant relationships
    n_samples = 1000
    
    # Create redundant sources
    common_signal = np.random.randn(n_samples)
    E = common_signal + 0.5 * np.random.randn(n_samples)
    I = common_signal + 0.5 * np.random.randn(n_samples)
    
    # Create target with synergistic component (XOR-like)
    synergy_component = ((E > 0) ^ (I > 0)).astype(float)
    C = (synergy_component + 0.1 * np.random.randn(n_samples) > 0.5).astype(int)
    
    # Output depends on both
    Vout = E + I + synergy_component
    
    # Configure for PID analysis
    config = InformationAnalysisPreset.pid_analysis(method="kraskov")
    analyzer = IntegratedInformationAnalyzer(config)
    
    results = analyzer.analyze(E, I, Vout, C)
    
    print("PID Results:")
    if "pid" in results:
        for pid_type, metrics in results["pid"].items():
            print(f"\n{pid_type}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    return results


def example_batch_analysis():
    """Example of analyzing multiple datasets/conditions."""
    print("\n=== Batch Analysis Example ===\n")
    
    # Different experimental conditions
    conditions = {
        "low_noise": 0.1,
        "medium_noise": 0.5,
        "high_noise": 1.0,
    }
    
    # Use the same configuration for all
    config = ConfigBuilder() \
        .with_method("sklearn") \
        .with_metrics(basic=True, pairwise=True, conditional=False, pid=False) \
        .with_sklearn_params(n_bins=20) \
        .build()
    
    analyzer = IntegratedInformationAnalyzer(config)
    
    results_by_condition = {}
    
    for condition_name, noise_level in conditions.items():
        print(f"\nAnalyzing {condition_name} (noise={noise_level})...")
        
        # Generate data with varying noise
        n_samples = 500
        E = np.random.randn(n_samples)
        I = np.random.randn(n_samples)
        signal = 0.7 * E - 0.3 * I
        Vout = signal + noise_level * np.random.randn(n_samples)
        C = (signal > 0).astype(int)
        
        # Analyze
        results = analyzer.analyze(E, I, Vout, C)
        results_by_condition[condition_name] = results
        
        # Report key metric
        mi_vout_c = results["basic_mi"]["I(Vout;C)"]
        print(f"  I(Vout;C) = {mi_vout_c:.4f}")
    
    return results_by_condition


def run_all_examples():
    """Run all examples."""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Method Comparison", example_method_comparison),
        ("Custom Configuration", example_custom_configuration),
        ("Conditional MI Analysis", example_conditional_mi_analysis),
        ("PID Analysis", example_pid_analysis),
        ("Batch Analysis", example_batch_analysis),
    ]
    
    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_all_examples() 