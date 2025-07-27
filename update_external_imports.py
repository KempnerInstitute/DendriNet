#!/usr/bin/env python3
"""
Script to update external references to the old plotting structure.
"""

import os
import re

# Mapping of old imports to new imports
IMPORT_MAPPING = {
    'from dendritic_modeling.plotting.analysis import': 'from dendritic_modeling.plotting.visualizations.analysis import',
    'from dendritic_modeling.plotting.network import': 'from dendritic_modeling.plotting.visualizations.network import',
    'from dendritic_modeling.plotting.training import': 'from dendritic_modeling.plotting.visualizations.training import',
    'from dendritic_modeling.plotting.utils import': 'from dendritic_modeling.plotting.core.utils import',
    'from dendritic_modeling.plotting.correlation_plots import': 'from dendritic_modeling.plotting.visualizations.correlation_plots import',
}

def find_files_with_plotting_imports():
    """Find all Python files that import from the old plotting structure."""
    files_to_update = []
    
    for root, dirs, files in os.walk('src/'):
        # Skip the plotting directory itself
        if 'plotting' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check if it has any old plotting imports
                    for old_import in IMPORT_MAPPING.keys():
                        if old_import in content:
                            files_to_update.append(file_path)
                            break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return list(set(files_to_update))  # Remove duplicates

def update_file_imports(file_path):
    """Update imports in a single file."""
    print(f"Updating {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all mapping changes
        for old_import, new_import in IMPORT_MAPPING.items():
            content = content.replace(old_import, new_import)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated {file_path}")
        else:
            print(f"  ℹ️  No changes needed in {file_path}")
            
    except Exception as e:
        print(f"  ❌ Error updating {file_path}: {e}")

def main():
    print("=== UPDATING EXTERNAL PLOTTING IMPORTS ===\n")
    
    # Find files that need updating
    files_to_update = find_files_with_plotting_imports()
    print(f"Found {len(files_to_update)} files that need updating:\n")
    
    for file_path in files_to_update:
        print(f"  - {file_path}")
    
    print(f"\n{'='*50}")
    
    # Update each file
    for file_path in files_to_update:
        update_file_imports(file_path)
    
    print(f"\n✨ Completed updating {len(files_to_update)} files!")

if __name__ == "__main__":
    main() 