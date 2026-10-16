"""
Data preprocessing for HAM10000 and ISIC 2019 datasets
Includes label unification, data cleaning, resizing, normalization, and augmentation
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from collections import Counter
import config


class SkinLesionDataset(Dataset):
    """Dataset class for skin lesion images"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = self.transform(image)
        
        return image, label


class DataPreprocessor:
    """Handles data preprocessing for skin lesion datasets"""
    
    def __init__(self):
        self.class_to_idx = config.CLASS_TO_IDX
        self.idx_to_class = config.IDX_TO_CLASS
        
    def get_augmentation_transforms(self, is_training=True):
        """Get data augmentation transforms"""
        if is_training:
            transform = A.Compose([
                A.Resize(config.INPUT_SIZE[0], config.INPUT_SIZE[1]),
                A.HorizontalFlip(p=config.AUGMENTATION_PROB),
                A.VerticalFlip(p=config.AUGMENTATION_PROB),
                A.RandomRotate90(p=config.AUGMENTATION_PROB),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=config.AUGMENTATION_PROB
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=config.AUGMENTATION_PROB
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(config.INPUT_SIZE[0], config.INPUT_SIZE[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        return transform
    
    def load_ham10000(self, data_dir):
        """Load HAM10000 dataset"""
        image_paths = []
        labels = []
        
        # Look for metadata file
        metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            
            # Images are in two directories: part_1 and part_2
            image_dir_part1 = os.path.join(data_dir, "HAM10000_images_part_1")
            image_dir_part2 = os.path.join(data_dir, "HAM10000_images_part_2")
            
            for _, row in df.iterrows():
                image_id = row['image_id']
                dx = row['dx'].lower()
                
                # Map to unified class
                if dx in config.CLASS_MAPPING['HAM10000']:
                    unified_class = config.CLASS_MAPPING['HAM10000'][dx]
                    
                    # Try part_1 first, then part_2
                    image_path = os.path.join(image_dir_part1, f"{image_id}.jpg")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(image_dir_part2, f"{image_id}.jpg")
                    
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        labels.append(self.class_to_idx[unified_class])
        else:
            # If no metadata, try to load from directory structure
            image_dir_part1 = os.path.join(data_dir, "HAM10000_images_part_1")
            image_dir_part2 = os.path.join(data_dir, "HAM10000_images_part_2")
            
            # Load from part_1
            if os.path.exists(image_dir_part1):
                for img_file in os.listdir(image_dir_part1):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Try to infer class from filename or use default
                        # Since we don't have class info without metadata, skip this path
                        pass
            
            # Load from part_2
            if os.path.exists(image_dir_part2):
                for img_file in os.listdir(image_dir_part2):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Try to infer class from filename or use default
                        # Since we don't have class info without metadata, skip this path
                        pass
        
        return image_paths, labels
    
    def load_isic2019(self, data_dir):
        """Load ISIC 2019 dataset"""
        image_paths = []
        labels = []
        
        # Load directly from class folders (AK, BCC, BKL, DF, MEL, NV, SCC, VASC)
        # This is more efficient than using the CSV file
        class_folders = {
            'AK': 'AKIEC',  # Actinic keratoses -> AKIEC
            'BCC': 'BCC',
            'BKL': 'BKL',
            'DF': 'DF',
            'MEL': 'MEL',
            'NV': 'NV',
            'SCC': 'SCC',
            'VASC': 'VASC'
        }
        
        for folder_name, unified_class in class_folders.items():
            class_dir = os.path.join(data_dir, folder_name)
            if os.path.exists(class_dir) and os.path.isdir(class_dir):
                if unified_class in self.class_to_idx:
                    class_idx = self.class_to_idx[unified_class]
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(class_dir, img_file)
                            image_paths.append(image_path)
                            labels.append(class_idx)
        
        # Also try loading from CSV if class folders don't have all images
        metadata_path = os.path.join(data_dir, "ISIC_2019_Training_GroundTruth.csv")
        if os.path.exists(metadata_path) and len(image_paths) == 0:
            df = pd.read_csv(metadata_path)
            image_dir = os.path.join(data_dir, "ISIC_2019_Training_Input")
            
            # If Training_Input doesn't exist, try loading from class folders using CSV mapping
            if not os.path.exists(image_dir):
                # Use CSV to map images to classes, but load from class folders
                for _, row in df.iterrows():
                    image_id = row['image']
                    # Find the class (one-hot encoded)
                    for class_name in config.UNIFIED_CLASSES:
                        if class_name in row and row[class_name] == 1.0:
                            # Try to find in class folders
                            folder_name = class_name
                            if class_name == 'AKIEC':
                                folder_name = 'AK'
                            
                            class_dir = os.path.join(data_dir, folder_name)
                            image_path = os.path.join(class_dir, f"{image_id}.jpg")
                            if not os.path.exists(image_path):
                                image_path = os.path.join(class_dir, f"{image_id}.png")
                            
                            if os.path.exists(image_path):
                                image_paths.append(image_path)
                                labels.append(self.class_to_idx[class_name])
                            break
        
        return image_paths, labels
    
    def combine_datasets(self):
        """Combine HAM10000 and ISIC 2019 datasets"""
        print("Loading HAM10000 dataset...")
        ham_paths, ham_labels = self.load_ham10000(config.HAM10000_DIR)
        print(f"Loaded {len(ham_paths)} images from HAM10000")
        
        print("Loading ISIC 2019 dataset...")
        isic_paths, isic_labels = self.load_isic2019(config.ISIC2019_DIR)
        print(f"Loaded {len(isic_paths)} images from ISIC 2019")
        
        # Combine
        all_paths = ham_paths + isic_paths
        all_labels = ham_labels + isic_labels
        
        print(f"Total combined images: {len(all_paths)}")
        
        # Print class distribution
        label_counts = Counter(all_labels)
        print("\nClass distribution:")
        for idx, count in sorted(label_counts.items()):
            print(f"  {config.IDX_TO_CLASS[idx]}: {count}")
        
        return all_paths, all_labels
    
    def split_incremental_tasks(self, image_paths, labels, num_tasks=4):
        """Split data into incremental tasks for class-incremental learning"""
        # Group by class
        class_data = {cls_idx: [] for cls_idx in range(len(config.UNIFIED_CLASSES))}
        for path, label in zip(image_paths, labels):
            class_data[label].append((path, label))
        
        # Split classes into tasks
        classes_per_task = len(config.UNIFIED_CLASSES) // num_tasks
        tasks = []
        
        for task_id in range(num_tasks):
            task_classes = list(range(
                task_id * classes_per_task,
                min((task_id + 1) * classes_per_task, len(config.UNIFIED_CLASSES))
            ))
            
            task_data = []
            for cls_idx in task_classes:
                task_data.extend(class_data[cls_idx])
            
            if task_data:
                task_paths, task_labels = zip(*task_data)
                tasks.append({
                    'task_id': task_id,
                    'classes': task_classes,
                    'paths': list(task_paths),
                    'labels': list(task_labels)
                })
                print(f"Task {task_id}: {len(task_paths)} images, classes {task_classes}")
        
        return tasks
    
    def create_data_loaders(self, image_paths, labels, batch_size=config.BATCH_SIZE, 
                           train_ratio=0.8, shuffle=True):
        """Create train and validation data loaders"""
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=1-train_ratio, 
            stratify=labels, random_state=42, shuffle=shuffle
        )
        
        # Create datasets
        train_transform = self.get_augmentation_transforms(is_training=True)
        val_transform = self.get_augmentation_transforms(is_training=False)
        
        train_dataset = SkinLesionDataset(train_paths, train_labels, train_transform, is_training=True)
        val_dataset = SkinLesionDataset(val_paths, val_labels, val_transform, is_training=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader


if __name__ == "__main__":
    # Test data preprocessing
    preprocessor = DataPreprocessor()
    image_paths, labels = preprocessor.combine_datasets()
    
    if len(image_paths) > 0:
        tasks = preprocessor.split_incremental_tasks(image_paths, labels, num_tasks=4)
        print(f"\nCreated {len(tasks)} incremental tasks")
    else:
        print("No images found. Please ensure datasets are in the correct directories.")

