import sys
import subprocess
import json
import os
from pathlib import Path

def install_kernel():
    """Install the data analysis agent kernel for Jupyter."""
    kernel_name = "data_analysis_agent"
    display_name = "Data Analysis Agent"
    
    # Create kernel specification directory
    kernel_dir = Path.home() / ".local" / "share" / "jupyter" / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the path to the current Python interpreter
    python_executable = sys.executable
    
    # Create kernel.json
    kernel_json = {
        "argv": [
            python_executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        ],
        "display_name": display_name,
        "language": "python"
    }
    
    # Write kernel.json
    with open(kernel_dir / "kernel.json", "w") as f:
        json.dump(kernel_json, f, indent=2)
    
    print(f"Kernel '{display_name}' installed successfully.")
    print(f"Kernel specification directory: {kernel_dir}")
    
    # Make sure ipykernel is installed
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "ipykernel"])
    except subprocess.CalledProcessError:
        print("Error installing ipykernel. Please install it manually.")
    
    # Install the package in development mode
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "-e", "."])
        print("Package installed in development mode.")
    except subprocess.CalledProcessError:
        print("Error installing package. Please install it manually with 'pip install -e .'")

if __name__ == "__main__":
    install_kernel()