#!/usr/bin/env python
"""
Unified Result Analyzer for Dendritic Modeling Framework
=======================================================

This script provides unified analysis capabilities for all sweep types.
It can automatically detect the sweep type and apply appropriate analysis.

Usage:
    # Analyze specific sweep type
    python result_analyzer.py --sweep-type ei --results-dir output/ei_20240101120000
    
    # Auto-detect sweep type
    python result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect
    
    # Analyze multiple directories
    python result_analyzer.py --results-dirs output/ei_* --sweep-type ei
"""

import argparse
import sys
import os
import glob
from pathlib import Path
from typing import Optional, List

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf
from dendritic_modeling.sweeps.unified.sweep_manager import SweepManager


class UnifiedResultAnalyzer:
    """
    Unified analyzer that can handle results from any sweep type.
    """
    
    def __init__(self):
        self.supported_types = SweepManager.SWEEP_TYPES
    
    def detect_sweep_type(self, results_dir: str) -> Optional[str]:
        """
        Auto-detect the sweep type from the results directory.
        
        Args:
            results_dir: Directory containing sweep results
            
        Returns:
            Detected sweep type or None if unable to detect
        """
        # Check for metadata file
        metadata_path = os.path.join(results_dir, "configs", "sweep_metadata.yaml")
        if os.path.exists(metadata_path):
            try:
                metadata = OmegaConf.load(metadata_path)
                return metadata.get("sweep_type")
            except Exception:
                pass
        
        # Check directory name patterns
        dir_name = os.path.basename(results_dir.rstrip('/'))
        for sweep_type in self.supported_types:
            if dir_name.startswith(f"{sweep_type}_"):
                return sweep_type
        
        # Check config file patterns
        config_dir = os.path.join(results_dir, "configs")
        if os.path.exists(config_dir):
            for sweep_type in self.supported_types:
                config_files = glob.glob(os.path.join(config_dir, f"{sweep_type}_config_*.yaml"))
                if config_files:
                    return sweep_type
        
        return None
    
    def analyze_single_sweep(self, results_dir: str, sweep_type: Optional[str] = None,
                           output_dir: Optional[str] = None) -> dict:
        """
        Analyze a single sweep directory.
        
        Args:
            results_dir: Directory containing sweep results
            sweep_type: Type of sweep (auto-detected if None)
            output_dir: Output directory for analysis (defaults to results_dir)
            
        Returns:
            Analysis results dictionary
        """
        # Auto-detect sweep type if not provided
        if sweep_type is None:
            sweep_type = self.detect_sweep_type(results_dir)
            if sweep_type is None:
                raise ValueError(f"Could not detect sweep type for {results_dir}")
            print(f"Auto-detected sweep type: {sweep_type}")
        
        # Create sweep manager for the detected type
        manager = SweepManager(sweep_type)
        
        # Analyze results
        analysis_info = manager.analyze_results(results_dir, output_dir)
        
        return analysis_info
    
    def analyze_multiple_sweeps(self, results_dirs: List[str], sweep_type: Optional[str] = None) -> List[dict]:
        """
        Analyze multiple sweep directories.
        
        Args:
            results_dirs: List of directories containing sweep results
            sweep_type: Type of sweep (auto-detected if None)
            
        Returns:
            List of analysis results dictionaries
        """
        analysis_results = []
        
        for results_dir in results_dirs:
            if not os.path.exists(results_dir):
                print(f"Warning: Directory does not exist: {results_dir}")
                continue
            
            try:
                print(f"\nAnalyzing: {results_dir}")
                analysis_info = self.analyze_single_sweep(results_dir, sweep_type)
                analysis_results.append(analysis_info)
                print(f"Analysis complete for {results_dir}")
                
            except Exception as e:
                print(f"❌ Error analyzing {results_dir}: {e}")
                continue
        
        return analysis_results
    
    def generate_comparison_report(self, analysis_results: List[dict], output_file: str):
        """
        Generate a comparison report across multiple sweep analyses.
        
        Args:
            analysis_results: List of analysis results from multiple sweeps
            output_file: Path to save the comparison report
        """
        with open(output_file, 'w') as f:
            f.write("Unified Sweep Analysis Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            for i, analysis in enumerate(analysis_results):
                f.write(f"Analysis {i+1}: {analysis['sweep_type']} Sweep\n")
                f.write("-" * 30 + "\n")
                f.write(f"Results Directory: {analysis['results_dir']}\n")
                f.write(f"Output Directory: {analysis['output_dir']}\n")
                f.write(f"Number of Plots: {len(analysis['plot_paths'])}\n")
                f.write(f"Report Path: {analysis['report_path']}\n\n")
                
                # Add best configurations if available
                if 'results' in analysis and 'configs' in analysis['results']:
                    f.write(f"Number of Configurations: {len(analysis['results']['configs'])}\n")
                
                f.write("\n")
        
        print(f"Comparison report saved to: {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Result Analyzer for Dendritic Modeling Sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific sweep type
  python result_analyzer.py --sweep-type ei --results-dir output/ei_20240101120000
  
  # Auto-detect sweep type
  python result_analyzer.py --results-dir output/ei_20240101120000 --auto-detect
  
  # Analyze multiple directories
  python result_analyzer.py --results-dirs output/ei_* --sweep-type ei
  
  # Generate comparison report
  python result_analyzer.py --results-dirs output/*_20240101* --auto-detect --comparison-report comparison.txt
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results-dir",
        help="Single directory containing sweep results to analyze"
    )
    input_group.add_argument(
        "--results-dirs",
        nargs="+",
        help="Multiple directories containing sweep results to analyze"
    )
    
    # Sweep type options
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--sweep-type",
        choices=SweepManager.SWEEP_TYPES,
        help="Type of sweep (required if not using --auto-detect)"
    )
    type_group.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect sweep type from directory contents"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        help="Output directory for analysis (defaults to results directory)"
    )
    parser.add_argument(
        "--comparison-report",
        help="Generate comparison report and save to this file (for multiple directories)"
    )
    
    return parser.parse_args()


def expand_glob_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns in directory list."""
    expanded = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            matches = glob.glob(pattern)
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    
    # Remove duplicates and sort
    return sorted(list(set(expanded)))


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        analyzer = UnifiedResultAnalyzer()
        
        if args.results_dir:
            # Single directory analysis
            if not args.sweep_type and not args.auto_detect:
                print("❌ Error: Must specify --sweep-type or --auto-detect for single directory")
                sys.exit(1)
            
            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = [analyzer.analyze_single_sweep(
                args.results_dir, sweep_type, args.output_dir
            )]
        
        else:
            # Multiple directories analysis
            if not args.sweep_type and not args.auto_detect:
                print("❌ Error: Must specify --sweep-type or --auto-detect for multiple directories")
                sys.exit(1)
            
            # Expand glob patterns
            results_dirs = expand_glob_patterns(args.results_dirs)
            
            if not results_dirs:
                print("❌ Error: No matching directories found")
                sys.exit(1)
            
            print(f"Found {len(results_dirs)} directories to analyze:")
            for dir_path in results_dirs:
                print(f"   {dir_path}")
            
            sweep_type = args.sweep_type if not args.auto_detect else None
            analysis_results = analyzer.analyze_multiple_sweeps(results_dirs, sweep_type)
        
        # Generate comparison report if requested
        if args.comparison_report and len(analysis_results) > 1:
            analyzer.generate_comparison_report(analysis_results, args.comparison_report)
        
        print(f"\nAnalysis complete! Processed {len(analysis_results)} sweep(s)")
        
        # Summary
        for analysis in analysis_results:
            print(f"{analysis['sweep_type']} sweep: {len(analysis['plot_paths'])} plots generated")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 