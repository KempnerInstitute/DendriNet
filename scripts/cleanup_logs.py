#!/usr/bin/env python3
"""
Utility script to clean up log files from the root directory.
Moves them to a logs/ directory for better organization.
"""

import os
import shutil
from datetime import datetime

def cleanup_root_logs():
    """Move log files from root directory to logs/ directory."""
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create archived logs subdirectory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = os.path.join(logs_dir, f'archived_{timestamp}')
    os.makedirs(archive_dir, exist_ok=True)
    
    # List of log files to move
    log_files = [
        'dendritic_modeling.log',
        'train_output.log',
        'train_output_final.log',
        'pytest_output.log'
    ]
    
    # Also find any other .log files in root
    for file in os.listdir(project_root):
        if file.endswith('.log') and file not in log_files:
            log_files.append(file)
    
    moved_files = []
    
    for log_file in log_files:
        src_path = os.path.join(project_root, log_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(archive_dir, log_file)
            try:
                shutil.move(src_path, dst_path)
                moved_files.append(log_file)
                print(f"✓ Moved {log_file} to {os.path.relpath(dst_path, project_root)}")
            except Exception as e:
                print(f"✗ Failed to move {log_file}: {e}")
    
    if moved_files:
        print(f"\nMoved {len(moved_files)} log files to {os.path.relpath(archive_dir, project_root)}")
        
        # Create a README in the archive directory
        readme_path = os.path.join(archive_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(f"Log files archived on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"These files were moved from the project root directory.\n\n")
            f.write("Files in this archive:\n")
            for file in moved_files:
                f.write(f"- {file}\n")
    else:
        print("No log files found in root directory to clean up.")
        # Remove empty archive directory
        try:
            os.rmdir(archive_dir)
        except:
            pass
    
    print("\nNote: Future logs will be saved in experiment-specific directories or the logs/ directory.")
    print("The dendritic_modeling.log file will now be saved in logs/ by default.")

if __name__ == "__main__":
    cleanup_root_logs() 