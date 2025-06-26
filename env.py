"""
Face Recognition System - Environment Setup
This script will check for required libraries and install them if needed.
"""
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    required_packages = [
        'opencv-python',
        'dlib',  # Note: dlib may require additional setup on Windows
        'face-recognition',
        'tensorflow',
        'numpy',
        'pandas',
        'tqdm',
        'pillow',
        'PyQt5'
    ]
    
    print("Setting up environment for Face Recognition System...")
    
    for package in required_packages:
        try:
            if package == 'face-recognition':
                # Check for face_recognition package
                __import__('face_recognition')
            else:
                # Check for other packages
                __import__(package.replace('-', '_'))
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"⏳ Installing {package}...")
            install_package(package)
            print(f"✅ {package} installed successfully")
    
    print("\nEnvironment setup complete! You're ready to start building your face recognition system.")

if __name__ == "__main__":
    setup_environment()