#!/usr/bin/env python3
"""
Script to identify unused functions and classes in the plotting directory.
This script analyzes actual usage patterns across the entire codebase.
"""

import os
import re
import subprocess
from collections import defaultdict

def extract_functions_and_classes(file_path):
    """Extract all function and class definitions from a Python file."""
    items = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find class definitions
        class_matches = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        for class_name in class_matches:
            items.append(('class', class_name))
        
        # Find function definitions
        func_matches = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        for func_name in func_matches:
            items.append(('function', func_name))
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return items

def check_usage(name, plotting_dir="src/dendritic_modeling/plotting/"):
    """Check if a function/class is used outside of the plotting directory."""
    try:
        # Search for usage outside plotting directory
        cmd = [
            "grep", "-r", name, "src/", 
            "--include=*.py",
            "--exclude-dir=__pycache__"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Filter out lines from plotting directory itself and __init__.py exports
        external_usage = []
        for line in lines:
            if line:
                file_path = line.split(':')[0]
                # Skip if it's in plotting directory (except managers which might use it)
                if plotting_dir in file_path and 'managers' not in file_path:
                    continue
                # Skip __init__.py exports unless it's used somewhere else
                if '__init__.py' in file_path:
                    continue
                external_usage.append(line)
        
        return len(external_usage) > 0, external_usage
    except Exception as e:
        print(f"Error checking usage for {name}: {e}")
        return False, []

def main():
    plotting_dir = "src/dendritic_modeling/plotting/"
    
    print("=== ANALYZING PLOTTING DIRECTORY FOR UNUSED FUNCTIONS/CLASSES ===\n")
    
    # Get all Python files in plotting directory
    python_files = []
    for root, dirs, files in os.walk(plotting_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                python_files.append(os.path.join(root, file))
    
    all_items = {}
    unused_items = defaultdict(list)
    used_items = defaultdict(list)
    
    # Analyze each file
    for file_path in python_files:
        print(f"Analyzing {file_path}...")
        items = extract_functions_and_classes(file_path)
        all_items[file_path] = items
        
        for item_type, name in items:
            # Skip private functions/classes and special methods
            if name.startswith('_'):
                continue
                
            is_used, usage_lines = check_usage(name)
            
            if is_used:
                used_items[file_path].append((item_type, name, len(usage_lines)))
            else:
                unused_items[file_path].append((item_type, name))
    
    # Print results
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    total_items = 0
    total_unused = 0
    
    for file_path in python_files:
        items = all_items.get(file_path, [])
        unused = unused_items.get(file_path, [])
        used = used_items.get(file_path, [])
        
        total_items += len(items)
        total_unused += len(unused)
        
        print(f"\n📁 {file_path}")
        print(f"   Total items: {len(items)}")
        print(f"   Used items: {len(used)}")
        print(f"   Unused items: {len(unused)}")
        
        if unused:
            print("   🗑️  UNUSED:")
            for item_type, name in unused:
                print(f"      - {item_type}: {name}")
        
        if used:
            print("   ✅ USED:")
            for item_type, name, usage_count in used[:5]:  # Show first 5
                print(f"      - {item_type}: {name} ({usage_count} references)")
            if len(used) > 5:
                print(f"      ... and {len(used) - 5} more")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total functions/classes analyzed: {total_items}")
    print(f"   Unused items: {total_unused}")
    print(f"   Usage rate: {((total_items - total_unused) / total_items * 100):.1f}%")
    
    # Create lists for easy removal
    print(f"\n🎯 FUNCTIONS/CLASSES TO REMOVE:")
    removal_candidates = []
    for file_path, unused in unused_items.items():
        if unused:
            print(f"\n   {file_path}:")
            for item_type, name in unused:
                print(f"      {name}")
                removal_candidates.append((file_path, item_type, name))
    
    print(f"\n✨ Total items recommended for removal: {len(removal_candidates)}")
    
    return unused_items

if __name__ == "__main__":
    main() 