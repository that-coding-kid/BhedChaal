#!/usr/bin/env python
"""
Helper script to add images to the README.md file
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Add image to README.md')
    parser.add_argument('image_file', help='Path to the image file (should be in assets folder)')
    parser.add_argument('description', help='Description for the image')
    parser.add_argument('--readme', default='../README.md', help='Path to README.md file')
    parser.add_argument('--table', action='store_true', help='Add to image table instead of inline')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image_file)
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found.")
        return 1
    
    # Get relative path to the image from the README location
    readme_dir = Path(args.readme).parent
    rel_path = os.path.relpath(image_path, readme_dir)
    
    # Format the markdown
    if args.table:
        markdown = f"| {args.description} | ![{args.description}]({rel_path}) |\n"
        print("\nTable row markdown:")
    else:
        markdown = f"![{args.description}]({rel_path})\n"
        print("\nInline image markdown:")
    
    print(markdown)
    
    # Check if README exists
    readme_path = Path(args.readme)
    if not readme_path.exists():
        print(f"Warning: README file '{readme_path}' not found. Not appending.")
        return 0
    
    # Ask for confirmation before modifying README
    choice = input(f"\nAppend this to {readme_path}? (y/n): ")
    if choice.lower() not in ['y', 'yes']:
        print("Not modifying README.")
        return 0
    
    # Append to README
    with open(readme_path, 'a') as f:
        f.write("\n")  # Add a new line
        f.write(markdown)
    
    print(f"Successfully added image reference to {readme_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 