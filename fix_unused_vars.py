#!/usr/bin/env python3
"""
Script to fix unused variable assignments in test files.
"""

import re
import os

def fix_unused_variables(file_path):
    """Fix unused variable assignments in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: model = ... (in test contexts where model is not used)
    content = re.sub(
        r'(\s+)model = (.*?)$',
        r'\1_ = \2',
        content,
        flags=re.MULTILINE
    )
    
    # Pattern 2: output = ... (in test contexts where output is not used)
    content = re.sub(
        r'(\s+)output = (.*?)$',
        r'\1_ = \2',
        content,
        flags=re.MULTILINE
    )
    
    # Pattern 3: x = ... (in test contexts where x is not used)
    content = re.sub(
        r'(\s+)x = (.*?)$',
        r'\1_ = \2',
        content,
        flags=re.MULTILINE
    )
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed unused variables in {file_path}")
    else:
        print(f"No changes needed in {file_path}")

def main():
    """Main function to fix unused variables in test files."""
    test_files = [
        "tests/test_dendrinet.py",
        "tests/test_stochastic_topk.py", 
        "tests/test_topklinear.py"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            fix_unused_variables(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main() 