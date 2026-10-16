"""
Training script for Class Incremental Continual Learning
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
import os
import json
from datetime import datetime
import config
from device_utils import (
    get_device, clear_cuda_cache, print_cuda_memory_info, 
    enable_mixed_precision, setup_cuda_environment, get_optimal_batch_size
)
from data_preprocessing import DataPreprocessor
from models.cicl_model import DenseNet201CICL, ExemplarMemory, CICLTrainer


def train_incremental_task(model, trainer, train_loader, val_loader, task_id, 
                          device, save_dir=config.MODEL_SAVE_DIR):
    """Train model on a single incremental task"""
    print(f"\n{'='*60}")
    print(f"Training Task {task_id}")
    print(f"{'='*60}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, config.EPOCHS_PER_TASK + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS_PER_TASK}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch, task_id)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, f"task_{task_id}_checkpoint.pth")
    torch.save({
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': best_val_acc,
    }, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    
    return best_val_acc


def evaluate(model, val_loader, device):
    """Evaluate model on validation set with CUDA optimization"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Use autocast for faster inference
    from torch.cuda.amp import autocast
    
    with torch.no_grad():
        for data, target in val_loader:
            # Use non_blocking for faster data transfer
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, target)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function"""
    print("Class Incremental Continual Learning for Skin Lesion Classification")
    print("="*70)
    
    # Device setup - Force CUDA usage
    print("\n" + "="*70)
    print("Device Configuration")
    print("="*70)
    setup_cuda_environment()
    device = get_device(force_cuda=True)  # Force CUDA, raise error if not available
    print_cuda_memory_info()
    
    # Adjust batch size based on GPU memory
    optimal_batch_size = get_optimal_batch_size(config.BATCH_SIZE, device)
    if optimal_batch_size != config.BATCH_SIZE:
        print(f"⚠ Adjusted batch size from {config.BATCH_SIZE} to {optimal_batch_size} based on GPU memory")
        config.BATCH_SIZE = optimal_batch_size
    
    # Enable mixed precision if available
    use_amp = enable_mixed_precision()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("✓ Mixed precision training enabled (faster training, lower memory)")
    
    # Data preprocessing
    print("\nPreprocessing data...")
    preprocessor = DataPreprocessor()
    image_paths, labels = preprocessor.combine_datasets()
    
    if len(image_paths) == 0:
        print("ERROR: No images found. Please ensure datasets are in:")
        print(f"  - {config.HAM10000_DIR}")
        print(f"  - {config.ISIC2019_DIR}")
        return
    
    # Split into incremental tasks
    num_tasks = 4
    tasks = preprocessor.split_incremental_tasks(image_paths, labels, num_tasks=num_tasks)
    
    # Initialize model
    model = DenseNet201CICL(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    model = model.to(device)
    
    # Initialize exemplar memory
    exemplar_memory = ExemplarMemory(memory_size=config.EXEMPLAR_MEMORY_SIZE)
    
    # Initialize trainer with mixed precision support
    trainer = CICLTrainer(model, device, exemplar_memory, use_amp=use_amp, scaler=scaler)
    
    # Training history
    training_history = {
        'task_accuracies': [],
        'final_accuracies': [],
        'forgetting_measures': []
    }
    
    previous_task_accuracies = {}
    
    # Train on each task incrementally
    for task_idx, task in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_idx + 1}/{len(tasks)}")
        print(f"Classes: {[config.IDX_TO_CLASS[c] for c in task['classes']]}")
        print(f"{'='*70}")
        
        # Create data loaders for this task
        train_loader, val_loader = preprocessor.create_data_loaders(
            task['paths'], task['labels'], 
            batch_size=config.BATCH_SIZE
        )
        
        # Compute Fisher Information Matrix for EWC (on previous task data)
        if task_idx > 0 and config.USE_REGULARIZATION:
            print("Computing Fisher Information Matrix for EWC...")
            # Use exemplar memory as proxy for previous task data
            exemplar_data, exemplar_labels = exemplar_memory.get_all_exemplars()
            if exemplar_data is not None:
                from torch.utils.data import TensorDataset, DataLoader
                exemplar_dataset = TensorDataset(exemplar_data, exemplar_labels)
                exemplar_loader = DataLoader(exemplar_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
                model.compute_fisher(exemplar_loader, device)
        
        # Set previous model for knowledge distillation
        if task_idx > 0 and config.USE_KNOWLEDGE_DISTILLATION:
            model.set_previous_model(model)
        
        # Train on current task
        task_acc = train_incremental_task(
            model, trainer, train_loader, val_loader, 
            task_idx, device
        )
        training_history['task_accuracies'].append(task_acc)
        
        # Clear CUDA cache periodically
        clear_cuda_cache()
        
        # Store exemplars for current task classes
        if config.USE_REPLAY:
            print("\nStoring exemplars for replay...")
            model.eval()
            for class_idx in task['classes']:
                # Get samples of this class
                class_samples = [(path, label) for path, label 
                                in zip(task['paths'], task['labels']) 
                                if label == class_idx]
                
                if len(class_samples) > 0:
                    # Sample some images for exemplar storage
                    import random
                    sample_size = min(50, len(class_samples))
                    sampled = random.sample(class_samples, sample_size)
                    
                    # Load and preprocess images
                    from PIL import Image
                    import numpy as np
                    transform = preprocessor.get_augmentation_transforms(is_training=False)
                    
                    images = []
                    labels_list = []
                    for path, label in sampled:
                        try:
                            img = Image.open(path).convert('RGB')
                            img = np.array(img)
                            transformed = transform(image=img)
                            images.append(transformed['image'])
                            labels_list.append(label)
                        except:
                            continue
                    
                    if len(images) > 0:
                        images_tensor = torch.stack(images)
                        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
                        exemplar_memory.add_exemplars(class_idx, images_tensor, labels_tensor, model, device)
            
            model.train()
        
        # Evaluate on all previous tasks
        if task_idx > 0:
            print("\nEvaluating on all previous tasks...")
            all_task_accuracies = {}
            for prev_task_idx in range(task_idx + 1):
                prev_task = tasks[prev_task_idx]
                _, prev_val_loader = preprocessor.create_data_loaders(
                    prev_task['paths'], prev_task['labels'],
                    batch_size=config.BATCH_SIZE
                )
                _, acc = evaluate(model, prev_val_loader, device)
                all_task_accuracies[prev_task_idx] = acc
                print(f"  Task {prev_task_idx} accuracy: {acc:.2f}%")
            
            # Compute forgetting measure
            if task_idx in previous_task_accuracies:
                forgetting = previous_task_accuracies[task_idx] - all_task_accuracies[task_idx]
                training_history['forgetting_measures'].append(forgetting)
                print(f"  Forgetting on task {task_idx}: {forgetting:.2f}%")
            
            previous_task_accuracies = all_task_accuracies
        
        training_history['final_accuracies'].append(task_acc)
    
    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    print(f"Task Accuracies: {training_history['task_accuracies']}")
    print(f"Average Accuracy: {sum(training_history['task_accuracies'])/len(training_history['task_accuracies']):.2f}%")
    if training_history['forgetting_measures']:
        print(f"Average Forgetting: {sum(training_history['forgetting_measures'])/len(training_history['forgetting_measures']):.2f}%")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

