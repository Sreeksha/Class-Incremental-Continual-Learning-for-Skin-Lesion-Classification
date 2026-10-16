"""
Test script to verify data loading works with the actual dataset structure
"""

import os
import config
from data_preprocessing import DataPreprocessor

def test_data_loading():
    """Test loading HAM10000 and ISIC 2019 datasets"""
    print("="*70)
    print("Testing Data Loading")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    
    # Test HAM10000 loading
    print("\n1. Testing HAM10000 dataset loading...")
    print(f"   Looking in: {config.HAM10000_DIR}")
    
    if not os.path.exists(config.HAM10000_DIR):
        print(f"   ERROR: Directory {config.HAM10000_DIR} not found!")
        return False
    
    ham_paths, ham_labels = preprocessor.load_ham10000(config.HAM10000_DIR)
    print(f"   ✓ Loaded {len(ham_paths)} images from HAM10000")
    
    if len(ham_paths) > 0:
        print(f"   Sample image: {ham_paths[0]}")
        print(f"   Sample label: {config.IDX_TO_CLASS[ham_labels[0]]}")
        
        # Count by class
        from collections import Counter
        label_counts = Counter(ham_labels)
        print("\n   HAM10000 class distribution:")
        for idx, count in sorted(label_counts.items()):
            print(f"     {config.IDX_TO_CLASS[idx]}: {count}")
    else:
        print("   WARNING: No images loaded from HAM10000!")
        return False
    
    # Test ISIC 2019 loading
    print("\n2. Testing ISIC 2019 dataset loading...")
    print(f"   Looking in: {config.ISIC2019_DIR}")
    
    if not os.path.exists(config.ISIC2019_DIR):
        print(f"   ERROR: Directory {config.ISIC2019_DIR} not found!")
        return False
    
    isic_paths, isic_labels = preprocessor.load_isic2019(config.ISIC2019_DIR)
    print(f"   ✓ Loaded {len(isic_paths)} images from ISIC 2019")
    
    if len(isic_paths) > 0:
        print(f"   Sample image: {isic_paths[0]}")
        print(f"   Sample label: {config.IDX_TO_CLASS[isic_labels[0]]}")
        
        # Count by class
        from collections import Counter
        label_counts = Counter(isic_labels)
        print("\n   ISIC 2019 class distribution:")
        for idx, count in sorted(label_counts.items()):
            print(f"     {config.IDX_TO_CLASS[idx]}: {count}")
    else:
        print("   WARNING: No images loaded from ISIC 2019!")
        return False
    
    # Test combining datasets
    print("\n3. Testing dataset combination...")
    all_paths, all_labels = preprocessor.combine_datasets()
    print(f"   ✓ Combined total: {len(all_paths)} images")
    
    if len(all_paths) > 0:
        from collections import Counter
        label_counts = Counter(all_labels)
        print("\n   Combined dataset class distribution:")
        for idx, count in sorted(label_counts.items()):
            print(f"     {config.IDX_TO_CLASS[idx]}: {count}")
        
        # Test splitting into tasks
        print("\n4. Testing incremental task splitting...")
        tasks = preprocessor.split_incremental_tasks(all_paths, all_labels, num_tasks=4)
        print(f"   ✓ Created {len(tasks)} incremental tasks")
        for task in tasks:
            print(f"     Task {task['task_id']}: {len(task['paths'])} images, "
                  f"classes: {[config.IDX_TO_CLASS[c] for c in task['classes']]}")
        
        print("\n" + "="*70)
        print("✓ All tests passed! Data loading is working correctly.")
        print("="*70)
        return True
    else:
        print("\n   ERROR: No images in combined dataset!")
        return False


if __name__ == "__main__":
    success = test_data_loading()
    if not success:
        print("\n✗ Some tests failed. Please check the dataset paths and structure.")
        exit(1)

