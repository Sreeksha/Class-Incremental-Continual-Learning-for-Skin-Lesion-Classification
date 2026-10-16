"""
Device utilities for CUDA/GPU support across different platforms
Handles device detection, memory management, and multi-GPU support
"""

import torch
import os


def get_device(device_id=None, force_cuda=True):
    """
    Get the best available device (CUDA preferred)
    
    Args:
        device_id: Specific CUDA device ID (0, 1, etc.) or None for auto
        force_cuda: If True, raise error if CUDA not available. If False, fallback to CPU
    
    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_id} not available. "
                               f"Only {torch.cuda.device_count()} devices found.")
            device = torch.device(f"cuda:{device_id}")
        else:
            # Use the first available CUDA device
            device = torch.device("cuda:0")
        
        # Set current device
        torch.cuda.set_device(device)
        
        print(f"✓ CUDA available: {torch.cuda.get_device_name(device)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ Using device: {device}")
        
        return device
    else:
        if force_cuda:
            raise RuntimeError(
                "CUDA is not available! This code requires a CUDA-capable GPU.\n"
                "Please ensure:\n"
                "1. You have a CUDA-capable GPU\n"
                "2. CUDA toolkit is installed\n"
                "3. PyTorch was installed with CUDA support\n"
                "   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
            )
        else:
            print("⚠ CUDA not available, falling back to CPU")
            return torch.device("cpu")


def get_device_count():
    """Get the number of available CUDA devices"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_cuda_memory_info(device=None):
    """Print CUDA memory information"""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        print(f"CUDA Memory - Allocated: {allocated:.2f} GB, "
              f"Reserved: {reserved:.2f} GB, "
              f"Total: {total:.2f} GB")


def enable_mixed_precision():
    """Check if mixed precision training is supported"""
    if torch.cuda.is_available():
        # Check if AMP is available (PyTorch 1.6+)
        return hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
    return False


def set_cudnn_benchmark(enabled=True):
    """Enable/disable cuDNN benchmark for faster training (if input sizes are constant)"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enabled
        if enabled:
            print("✓ cuDNN benchmark enabled (faster training with fixed input sizes)")


def set_cudnn_deterministic(enabled=False):
    """Enable/disable deterministic cuDNN operations (slower but reproducible)"""
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = enabled
        torch.use_deterministic_algorithms(enabled)
        if enabled:
            print("⚠ Deterministic mode enabled (may be slower)")


def get_optimal_batch_size(base_batch_size=32, device=None):
    """Get optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return base_batch_size
    
    if device is None:
        device = torch.cuda.current_device()
    
    # Get GPU memory in GB
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    # Adjust batch size based on GPU memory
    if total_memory >= 24:  # 24GB+ (RTX 3090, A100, etc.)
        return base_batch_size * 2
    elif total_memory >= 12:  # 12-24GB (RTX 3080, RTX 4070, etc.)
        return base_batch_size
    elif total_memory >= 8:  # 8-12GB (RTX 3070, RTX 4060, etc.)
        return max(base_batch_size // 2, 16)
    else:  # <8GB (GTX 1660, RTX 3050, etc.)
        return max(base_batch_size // 4, 8)


def move_to_device(tensor_or_model, device):
    """Safely move tensor or model to device"""
    if isinstance(tensor_or_model, torch.nn.Module):
        return tensor_or_model.to(device)
    elif isinstance(tensor_or_model, torch.Tensor):
        return tensor_or_model.to(device)
    elif isinstance(tensor_or_model, (list, tuple)):
        return type(tensor_or_model)(move_to_device(item, device) for item in tensor_or_model)
    elif isinstance(tensor_or_model, dict):
        return {k: move_to_device(v, device) for k, v in tensor_or_model.items()}
    else:
        return tensor_or_model


def setup_cuda_environment():
    """Setup optimal CUDA environment settings"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for faster training
        set_cudnn_benchmark(True)
        
        # Set memory fraction (optional, can be set via environment variable)
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print("✓ CUDA environment configured")
        return True
    return False


if __name__ == "__main__":
    # Test device detection
    print("="*70)
    print("CUDA Device Detection Test")
    print("="*70)
    
    try:
        device = get_device(force_cuda=False)
        print(f"\nSelected device: {device}")
        
        if torch.cuda.is_available():
            print_cuda_memory_info()
            print(f"\nNumber of CUDA devices: {get_device_count()}")
            print(f"Mixed precision available: {enable_mixed_precision()}")
            setup_cuda_environment()
        else:
            print("\n⚠ CUDA not available - using CPU")
    except Exception as e:
        print(f"\n✗ Error: {e}")
