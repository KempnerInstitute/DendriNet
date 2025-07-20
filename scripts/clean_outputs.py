#!/usr/bin/env python
"""
Clean old outputs and WandB directories to manage disk space.

This script removes directories older than a specified number of days
from outputs/ and .wandb/ directories to prevent disk space bloat.

Usage:
    python scripts/clean_outputs.py [--days 30] [--dry-run]
"""

import argparse
import os
import shutil
import time
from pathlib import Path


def clean_old_directories(base_dir: str, days_old: int = 30, dry_run: bool = False) -> int:
    """
    Clean directories older than specified days.
    
    Args:
        base_dir: Base directory to clean
        days_old: Remove directories older than this many days
        dry_run: If True, only print what would be removed
    
    Returns:
        Number of directories that were (or would be) removed
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist, skipping...")
        return 0
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)  # Convert days to seconds
    removed_count = 0
    total_size = 0
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning {base_dir}...")
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            # Get the modification time of the directory
            mod_time = os.path.getmtime(item_path)
            
            if mod_time < cutoff_time:
                # Calculate size before removal
                size = get_directory_size(item_path)
                total_size += size
                size_mb = size / (1024 * 1024)
                
                print(f"{'[DRY RUN] ' if dry_run else ''}Removing {item_path} ({size_mb:.1f} MB)")
                
                if not dry_run:
                    try:
                        shutil.rmtree(item_path)
                        removed_count += 1
                    except OSError as e:
                        print(f"Error removing {item_path}: {e}")
                else:
                    removed_count += 1
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"{'[DRY RUN] ' if dry_run else ''}Removed {removed_count} directories, "
          f"freed {total_size_mb:.1f} MB from {base_dir}")
    
    return removed_count


def get_directory_size(directory: str) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                # File might have been deleted or is inaccessible
                pass
    return total_size


def main():
    parser = argparse.ArgumentParser(description="Clean old outputs and WandB directories")
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Remove directories older than this many days (default: 30)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be removed without actually removing it"
    )
    
    args = parser.parse_args()
    
    # Directories to clean
    dirs_to_clean = [".wandb", "outputs"]
    
    total_removed = 0
    
    print(f"Cleaning directories older than {args.days} days...")
    if args.dry_run:
        print("DRY RUN MODE - No files will be actually removed")
    
    for directory in dirs_to_clean:
        removed = clean_old_directories(directory, args.days, args.dry_run)
        total_removed += removed
    
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total: {total_removed} directories processed")
    
    if not args.dry_run and total_removed > 0:
        print("Cleanup completed successfully!")
    elif args.dry_run:
        print("Run without --dry-run to actually remove the directories")


if __name__ == "__main__":
    main() 