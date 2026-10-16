# CUDA Setup and GPU Configuration

This project is **fully optimized for CUDA/GPU** and will not run on CPU. All operations are designed to utilize GPU acceleration.

## Features

✅ **Automatic CUDA Detection**: Automatically detects and uses available CUDA devices  
✅ **Multi-GPU Support**: Can specify which GPU to use (device 0, 1, etc.)  
✅ **Mixed Precision Training**: Uses FP16/FP32 mixed precision for faster training  
✅ **Memory Optimization**: Automatically adjusts batch size based on GPU memory  
✅ **CUDA Optimizations**: Non-blocking transfers, cuDNN benchmarking, memory management  

## Device Detection

The code uses `device_utils.py` which provides:

- `get_device()`: Automatically selects the best CUDA device
- `setup_cuda_environment()`: Configures optimal CUDA settings
- `get_optimal_batch_size()`: Adjusts batch size based on GPU memory
- `print_cuda_memory_info()`: Shows GPU memory usage

## GPU Memory Recommendations

| GPU Memory | Recommended Batch Size | Example GPUs |
|------------|------------------------|--------------|
| 24GB+ | 64 | RTX 3090, A100, RTX 4090 |
| 12-24GB | 32 | RTX 3080, RTX 4070, RTX 4080 |
| 8-12GB | 16 | RTX 3070, RTX 4060 Ti |
| <8GB | 8 | GTX 1660, RTX 3050 |

## CUDA Platform Compatibility

The code is compatible with:
- **NVIDIA GPUs** with Compute Capability 3.5+
- **CUDA 11.0+** (11.8+ recommended)
- **PyTorch** with CUDA support
- **Windows, Linux, macOS** (with NVIDIA GPU)

## Installation

1. **Install CUDA Toolkit** (if not already installed):
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Version 11.8 or 12.1 recommended

2. **Install PyTorch with CUDA**:
   ```bash
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify Installation**:
   ```bash
   python device_utils.py
   ```

## Configuration

In `config.py`, you can configure:

```python
FORCE_CUDA = True  # Set to False to allow CPU fallback (not recommended)
CUDA_DEVICE_ID = None  # None for auto, or 0, 1, etc. for specific GPU
USE_MIXED_PRECISION = True  # Enable FP16/FP32 mixed precision
BATCH_SIZE = 32  # Will be auto-adjusted based on GPU memory
```

## Performance Optimizations

The code includes several CUDA optimizations:

1. **Mixed Precision Training**: Uses `torch.cuda.amp` for faster training
2. **Non-blocking Transfers**: Uses `non_blocking=True` for data loading
3. **cuDNN Benchmarking**: Enabled for faster convolutions
4. **Memory Management**: Automatic cache clearing and memory monitoring
5. **Gradient Scaling**: Proper gradient scaling for mixed precision

## Troubleshooting

### "CUDA is not available" Error

**Solution**: Install PyTorch with CUDA support:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM) Error

**Solutions**:
1. Reduce `BATCH_SIZE` in `config.py`
2. The code auto-adjusts batch size, but you can manually set it lower
3. Enable gradient checkpointing (if implemented)
4. Use smaller model or reduce input image size

### Slow Training

**Solutions**:
1. Ensure mixed precision is enabled (`USE_MIXED_PRECISION = True`)
2. Check that cuDNN benchmarking is enabled (automatic)
3. Use larger batch size if GPU memory allows
4. Ensure data loading uses multiple workers

## Testing CUDA Setup

Run the device utility test:
```bash
python device_utils.py
```

Expected output:
```
======================================================================
CUDA Device Detection Test
======================================================================
✓ CUDA available: NVIDIA GeForce RTX 3090
✓ CUDA version: 11.8
✓ cuDNN version: 8500
✓ Using device: cuda:0
CUDA Memory - Allocated: 0.00 GB, Reserved: 0.00 GB, Total: 24.00 GB
Number of CUDA devices: 1
Mixed precision available: True
✓ CUDA environment configured
```

## Multi-GPU Support

To use a specific GPU:
```python
# In config.py
CUDA_DEVICE_ID = 0  # Use first GPU
CUDA_DEVICE_ID = 1  # Use second GPU
```

Or in code:
```python
from device_utils import get_device
device = get_device(device_id=0)  # Use GPU 0
device = get_device(device_id=1)  # Use GPU 1
```

## Notes

- The code **requires CUDA** and will raise an error if CUDA is not available
- All tensors and models are automatically moved to GPU
- Data loaders use `pin_memory=True` for faster CPU-GPU transfers
- Mixed precision training reduces memory usage and speeds up training

