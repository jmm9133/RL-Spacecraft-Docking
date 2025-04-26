import torch
import gc
import os

def clear_gpu_memory():
    """
    Function to clear PyTorch cache and release GPU memory
    """
    # Print initial GPU memory usage
    if torch.cuda.is_available():
        print("Initial GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB reserved")
    
    # Empty the cache
    torch.cuda.empty_cache()
    
    # Delete all variables in PyTorch
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    # Empty cache again after collection
    torch.cuda.empty_cache()
    
    # Reset the CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Print final GPU memory usage
        print("\nGPU Memory After Cleanup:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB reserved")

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s)")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available")
        
    # Clear the memory
    clear_gpu_memory()
    
    print("\nMemory cleanup completed!")