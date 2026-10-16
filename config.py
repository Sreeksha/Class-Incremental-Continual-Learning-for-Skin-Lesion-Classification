"""
Configuration file for Class Incremental Continual Learning for Skin Lesion Classification

NOTE: This code REQUIRES CUDA/GPU support. It will not run on CPU.
Make sure you have:
1. A CUDA-capable GPU (NVIDIA)
2. CUDA toolkit installed
3. PyTorch with CUDA support installed
   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import os

# Dataset paths
DATA_DIR = "__your dataset directory__"
HAM10000_DIR = "__your dataset directory__"  # HAM10000 dataset directory
ISIC2019_DIR = "__your dataset directory__"  # ISIC 2019 dataset directory
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Device configuration
FORCE_CUDA = True  # Set to False to allow CPU fallback
CUDA_DEVICE_ID = None  # None for auto-selection, or specify device ID (0, 1, etc.)
USE_MIXED_PRECISION = True  # Use mixed precision training for faster training

# Model configuration
MODEL_NAME = "densenet201"
PRETRAINED = True
NUM_CLASSES = 8  # Combined dataset classes
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32  # Adjust based on GPU memory (increase for larger GPUs)
LEARNING_RATE = 0.001
EPOCHS_PER_TASK = 50

# CICL Strategy configuration
USE_REPLAY = True
USE_REGULARIZATION = True
USE_KNOWLEDGE_DISTILLATION = True
EXEMPLAR_MEMORY_SIZE = 200  # Number of exemplars per class to store
REPLAY_RATIO = 0.3  # Ratio of replay samples in each batch

# Regularization parameters
LAMBDA_EWC = 0.4  # Elastic Weight Consolidation weight
LAMBDA_KD = 0.5  # Knowledge Distillation weight
TEMPERATURE = 4.0  # Temperature for knowledge distillation

# Data augmentation
AUGMENTATION_PROB = 0.5
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# Evaluation
EVAL_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'forgetting']

# Output paths
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"
LOG_DIR = "logs"

# Create directories
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Class mapping for label unification
CLASS_MAPPING = {
    'HAM10000': {
        'akiec': 'AKIEC',
        'bcc': 'BCC',
        'bkl': 'BKL',
        'df': 'DF',
        'mel': 'MEL',
        'nv': 'NV',
        'vasc': 'VASC'
    },
    'ISIC2019': {
        'AK': 'AKIEC',
        'BCC': 'BCC',
        'BKL': 'BKL',
        'DF': 'DF',
        'MEL': 'MEL',
        'NV': 'NV',
        'SCC': 'SCC',
        'VASC': 'VASC'
    }
}

# Unified class list
UNIFIED_CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(UNIFIED_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

