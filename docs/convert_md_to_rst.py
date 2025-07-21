#!/usr/bin/env python
"""
Simple converter from markdown to reStructuredText for Sphinx.
"""

import os
import glob
import re

def simple_md_to_rst(content):
    """Simple markdown to RST conversion."""
    # Convert headers
    content = re.sub(r'^# (.+)$', r'\1\n' + '=' * 50, content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'\1\n' + '-' * 30, content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'\1\n' + '~' * 20, content, flags=re.MULTILINE)
    content = re.sub(r'^#### (.+)$', r'\1\n' + '^' * 15, content, flags=re.MULTILINE)
    
    # Convert bold text
    content = re.sub(r'\*\*(.+?)\*\*', r'**\1**', content)
    
    # Convert italic text
    content = re.sub(r'\*(.+?)\*', r'*\1*', content)
    
    # Convert inline code
    content = re.sub(r'`(.+?)`', r'``\1``', content)
    
    # Convert code blocks
    content = re.sub(r'^```(\w+)?$', r'.. code-block:: \1', content, flags=re.MULTILINE)
    content = re.sub(r'^```$', r'', content, flags=re.MULTILINE)
    
    # Convert bullet lists (basic)
    content = re.sub(r'^- (.+)$', r'* \1', content, flags=re.MULTILINE)
    
    # Convert numbered lists (basic)
    content = re.sub(r'^\d+\. (.+)$', r'#. \1', content, flags=re.MULTILINE)
    
    return content

def convert_md_to_rst():
    """Convert all .md files in the current directory to .rst files."""
    md_files = glob.glob("*.md")
    
    print(f"Found {len(md_files)} markdown files to convert:")
    for md_file in md_files:
        print(f"  - {md_file}")
    
    converted_files = []
    
    for md_file in md_files:
        # Generate .rst filename
        rst_file = md_file.replace('.md', '.rst')
        
        try:
            # Read markdown file
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert to RST
            rst_content = simple_md_to_rst(md_content)
            
            # Write to .rst file
            with open(rst_file, 'w', encoding='utf-8') as f:
                f.write(rst_content)
            
            converted_files.append((md_file, rst_file))
            print(f"  ✓ Successfully converted {md_file}")
            
        except Exception as e:
            print(f"  ✗ Error converting {md_file}: {e}")
    
    # Move original .md files to backup directory
    backup_dir = "md_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    for md_file, rst_file in converted_files:
        try:
            os.rename(md_file, os.path.join(backup_dir, md_file))
            print(f"  ✓ Moved {md_file} to {backup_dir}/")
        except Exception as e:
            print(f"  ✗ Error moving {md_file}: {e}")
    
    print(f"\nConversion complete!")
    print(f"Converted {len(converted_files)} files")
    print(f"Original .md files backed up to {backup_dir}/")

if __name__ == "__main__":
    convert_md_to_rst()
