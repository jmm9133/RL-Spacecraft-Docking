import os
import sys
import subprocess

def run_command(command):
    """Run a shell command and return the output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print("Output:")
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print("Error:")
        print(result.stderr)
    print("-" * 50)
    return result

def check_nvidia_drivers():
    """Check if NVIDIA drivers are properly installed"""
    print("\n=== CHECKING NVIDIA DRIVERS ===")
    run_command("nvidia-smi")

def check_cuda_installation():
    """Check CUDA installation"""
    print("\n=== CHECKING CUDA INSTALLATION ===")
    run_command("nvcc --version")
    
    # Check CUDA path
    print("\nCUDA Path Variables:")
    for var in ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def check_pytorch_gpu():
    """Check PyTorch GPU detection"""
    print("\n=== CHECKING PYTORCH GPU DETECTION ===")
    print("Importing torch...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA is not available in PyTorch")
            
        # Check PyTorch CUDA configuration
        print("\nPyTorch CUDA Configuration:")
        if hasattr(torch, '_C'):
            print(f"Built with CUDA: {torch._C._built_with_cuda()}")
        print(f"CUDNN available: {torch.backends.cudnn.is_available()}")
        if torch.backends.cudnn.is_available():
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
    except ImportError:
        print("PyTorch is not installed")
    except Exception as e:
        print(f"Error during PyTorch check: {str(e)}")

def check_env_variables():
    """Check relevant environment variables"""
    print("\n=== CHECKING ENVIRONMENT VARIABLES ===")
    relevant_vars = ['PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'PYTHONPATH']
    for var in relevant_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

if __name__ == "__main__":
    print("GPU Diagnostic Tool for PyTorch")
    print("=" * 50)
    
    check_nvidia_drivers()
    check_cuda_installation()
    check_pytorch_gpu()
    check_env_variables()
    
    print("\n=== SYSTEM INFORMATION ===")
    print(f"Python version: {sys.version}")
    run_command("pip list | grep -E 'torch|cuda'")
