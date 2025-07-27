#!/usr/bin/env python3
"""
Script to systematically clean up the plotting directory by removing unused functions
and reorganizing the structure.
"""

import os
import re
import shutil

# Define the unused items identified by the analysis
UNUSED_ITEMS = {
    "src/dendritic_modeling/plotting/analysis.py": [
        "LayerAnalysisPlotter",
        "AblationPlotter", 
        "NoiseAnalysisPlotter",
        "WeightDistributionPlotter",
        "InformationAnalysisPlotter",
        "ActivationPlotter",
        "dendrinet_activations_todict",
        "einet_activations_todict"
    ],
    "src/dendritic_modeling/plotting/correlation_plots.py": [
        "plot_correlation_summary"
    ],
    "src/dendritic_modeling/plotting/network.py": [
        "plot_with_color",
        "add_colorbars", 
        "build_dendritic_tree_graph",
        "shape_helper",
        "dendrinet_params_todict",
        "einet_params_todict",
        "dendrinet_gradients_todict",
        "einet_gradients_todict"
    ],
    "src/dendritic_modeling/plotting/utils.py": [
        "reverse_layer_indices",
        "format_layer_labels",
        "dendrinet_activations_to_csv", 
        "einet_activations_to_csv",
        "build_dendritic_tree_graph",
        "split_mnist_inputs"
    ]
}

def find_item_boundaries(content, item_name, item_type="function"):
    """Find the start and end line numbers for a function or class."""
    lines = content.split('\n')
    start_line = None
    end_line = None
    indent_level = None
    
    # Find the definition line
    pattern = rf"^(class|def)\s+{re.escape(item_name)}\b"
    for i, line in enumerate(lines):
        if re.match(pattern, line):
            start_line = i
            indent_level = len(line) - len(line.lstrip())
            break
    
    if start_line is None:
        return None, None
    
    # Find the end by looking for the next item at the same or lesser indentation
    for i in range(start_line + 1, len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line at the same or lesser indentation that starts a new definition
        if current_indent <= indent_level and (line.strip().startswith('def ') or 
                                             line.strip().startswith('class ') or
                                             line.strip().startswith('# =')):
            end_line = i - 1
            break
    
    # If we didn't find an end, go to end of file
    if end_line is None:
        end_line = len(lines) - 1
    
    # Backtrack to remove trailing empty lines
    while end_line > start_line and lines[end_line].strip() == "":
        end_line -= 1
    
    return start_line, end_line

def remove_unused_items_from_file(file_path, unused_items):
    """Remove unused items from a file."""
    print(f"Cleaning {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    lines_to_remove = set()
    
    for item in unused_items:
        start_line, end_line = find_item_boundaries(content, item)
        if start_line is not None and end_line is not None:
            print(f"  Removing {item} (lines {start_line+1}-{end_line+1})")
            for line_num in range(start_line, end_line + 1):
                lines_to_remove.add(line_num)
        else:
            print(f"  Warning: Could not find {item}")
    
    # Create new content without the removed lines
    new_lines = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    new_content = '\n'.join(new_lines)
    
    # Remove excessive blank lines (more than 2 consecutive)
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  Removed {len(lines_to_remove)} lines from {file_path}")

def update_init_file():
    """Update the main __init__.py file to remove imports of deleted items."""
    init_file = "src/dendritic_modeling/plotting/__init__.py"
    print(f"Updating {init_file}...")
    
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove imports from deleted base.py
    content = re.sub(r'from dendritic_modeling\.plotting\.base import \([^)]+\)', '', content)
    
    # Remove unused imports from analysis.py
    unused_analysis_imports = [
        "AblationPlotter", "ActivationPlotter", "InformationAnalysisPlotter",
        "LayerAnalysisPlotter", "NoiseAnalysisPlotter", "WeightDistributionPlotter",
        "dendrinet_activations_todict", "einet_activations_todict"
    ]
    
    for item in unused_analysis_imports:
        content = re.sub(rf'\s*{re.escape(item)},?\n?', '', content)
    
    # Remove unused imports from other files
    unused_network_imports = [
        "dendrinet_gradients_todict", "dendrinet_params_todict", 
        "einet_gradients_todict", "einet_params_todict"
    ]
    
    for item in unused_network_imports:
        content = re.sub(rf'\s*{re.escape(item)},?\n?', '', content)
    
    # Remove items from __all__ list
    all_unused = (unused_analysis_imports + unused_network_imports + 
                 ["BarPlotter", "BasePlotter", "BoxPlotter", "ComparisonPlotter", 
                  "LinePlotter", "PlotConfig"])
    
    for item in all_unused:
        content = re.sub(rf'\s*"{re.escape(item)}",?\n?', '', content)
    
    # Clean up formatting
    content = re.sub(r',\s*\)', ')', content)  # Remove trailing commas
    content = re.sub(r'\n\n+', '\n\n', content)  # Remove excessive blank lines
    
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  Updated __init__.py")

def main():
    print("=== STARTING SYSTEMATIC CLEANUP OF PLOTTING DIRECTORY ===\n")
    
    # Remove unused items from each file
    for file_path, unused_items in UNUSED_ITEMS.items():
        if os.path.exists(file_path):
            remove_unused_items_from_file(file_path, unused_items)
        else:
            print(f"Warning: {file_path} not found")
    
    # Update the main __init__.py file
    update_init_file()
    
    print("\n=== CLEANUP COMPLETED ===")
    print("Next steps:")
    print("1. Test imports: python -c 'from dendritic_modeling.plotting import *'")
    print("2. Create subdirectory structure")
    print("3. Move files to subdirectories")
    print("4. Update import paths")

if __name__ == "__main__":
    main() 