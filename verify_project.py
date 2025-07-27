#!/usr/bin/env python3
"""
Project Verification Script
Checks that all files are present and properly structured
"""

import os

def verify_project_structure():
    """Verify all project files are present"""
    print(" Verifying project structure...")

    required_files = [
        'main.py',
        'drowsiness_detector.py', 
        'person_detector.py',
        'age_predictor.py',
        'utils.py',
        'config.json',
        'requirements.txt',
        'setup.py',
        'test_system.py',
        'README.md'
    ]

    missing_files = []
    present_files = []

    for file in required_files:
        if os.path.exists(file):
            present_files.append(file)
            print(f" {file}")
        else:
            missing_files.append(file)
            print(f" {file} - MISSING")

    print(f"\n Summary:")
    print(f"Present: {len(present_files)}/{len(required_files)} files")

    if missing_files:
        print(f"\n Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n All project files present!")
        return True

def check_file_sizes():
    """Check if files have reasonable content"""
    print("\n Checking file sizes...")

    files_info = {}
    for filename in os.listdir('.'):
        if filename.endswith('.py') or filename.endswith('.json') or filename.endswith('.md') or filename.endswith('.txt'):
            size = os.path.getsize(filename)
            files_info[filename] = size
            print(f" {filename}: {size} bytes")

    return files_info

def create_directory_structure():
    """Create necessary directories"""
    print("\n Creating directory structure...")

    directories = ['models', 'outputs', 'test_videos', 'logs']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" Created: {directory}/")
        else:
            print(f" Exists: {directory}/")

def main():
    print(" Multi-Person Drowsiness Detection System - Project Verification")
    print("=" * 70)

    # Verify files
    files_ok = verify_project_structure()

    # Check file sizes
    file_sizes = check_file_sizes()

    # Create directories
    create_directory_structure()

    print("\n" + "=" * 70)

    if files_ok:
        print(" PROJECT VERIFICATION COMPLETE!")
        print("\n Next steps:")
        print("1. Run: python setup.py")
        print("2. Test: python test_system.py --quick") 
        print("3. Launch: python main.py")
        print("\n Ready to detect drowsiness without dlib!")
    else:
        print(" Project verification failed. Some files are missing.")

    return files_ok

if __name__ == "__main__":
    main()
