"""
Evaluation script for Class Incremental Continual Learning model
Computes overall accuracy, per-class metrics, and forgetting measure
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import config
from data_preprocessing import DataPreprocessor
from models.cicl_model import DenseNet201CICL


def evaluate_model(model, data_loader, device, class_names=None):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Use autocast for faster inference
    from torch.cuda.amp import autocast
    
    with torch.no_grad():
        for data, target in data_loader:
            # Use non_blocking for faster data transfer
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(data)
                probs = F.softmax(outputs, dim=1)
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall accuracy
    overall_acc = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(class_names) if class_names else len(np.unique(all_labels))):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    results = {
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': [float(acc) for acc in per_class_acc],
        'per_class_precision': [float(p) for p in precision],
        'per_class_recall': [float(r) for r in recall],
        'per_class_f1': [float(f) for f in f1],
        'per_class_support': [int(s) for s in support],
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    return results


def compute_forgetting_measure(accuracies_per_task):
    """Compute forgetting measure across tasks"""
    if len(accuracies_per_task) < 2:
        return {}
    
    forgetting = {}
    for task_id in range(len(accuracies_per_task) - 1):
        # Forgetting is the difference between peak accuracy and final accuracy
        task_accs = accuracies_per_task[task_id]
        if len(task_accs) > 0:
            peak_acc = max(task_accs)
            final_acc = task_accs[-1]
            forgetting[task_id] = peak_acc - final_acc
    
    return forgetting


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(results, class_names, save_path):
    """Plot per-class metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].bar(class_names, results['per_class_accuracy'])
    axes[0, 0].set_title('Per-Class Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision
    axes[0, 1].bar(class_names, results['per_class_precision'])
    axes[0, 1].set_title('Per-Class Precision')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1, 0].bar(class_names, results['per_class_recall'])
    axes[1, 0].set_title('Per-Class Recall')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F1 Score
    axes[1, 1].bar(class_names, results['per_class_f1'])
    axes[1, 1].set_title('Per-Class F1 Score')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CICL Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--task_id', type=int, default=None,
                       help='Evaluate on specific task (None for all tasks)')
    parser.add_argument('--output_dir', type=str, default=config.RESULTS_DIR,
                       help='Output directory for results')
    parser.add_argument('--non_strict', action='store_true',
                       help='Load checkpoint with strict=False (e.g. if it contains extra keys)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup - Force CUDA usage
    from device_utils import get_device, setup_cuda_environment, print_cuda_memory_info
    
    print("\n" + "="*70)
    print("Device Configuration")
    print("="*70)
    setup_cuda_environment()
    device = get_device(force_cuda=True)  # Force CUDA, raise error if not available
    print_cuda_memory_info()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = DenseNet201CICL(num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=not args.non_strict)
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    image_paths, labels = preprocessor.combine_datasets()
    
    if len(image_paths) == 0:
        print("ERROR: No images found.")
        return
    
    # Split into tasks
    tasks = preprocessor.split_incremental_tasks(image_paths, labels, num_tasks=4)
    
    # Class names
    class_names = config.UNIFIED_CLASSES
    
    # Evaluate
    all_results = {}
    
    if args.task_id is not None:
        # Evaluate on specific task
        task = tasks[args.task_id]
        _, val_loader = preprocessor.create_data_loaders(
            task['paths'], task['labels'],
            batch_size=config.BATCH_SIZE
        )
        
        print(f"\nEvaluating on Task {args.task_id}...")
        results = evaluate_model(model, val_loader, device, class_names)
        all_results[f'task_{args.task_id}'] = results
        
        print(f"\nOverall Accuracy: {results['overall_accuracy']*100:.2f}%")
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(class_names):
            try:
                print(f"  {class_name}:")
                print(f"    Accuracy: {results['per_class_accuracy'][i]*100:.2f}%")
                print(f"    Precision: {results['per_class_precision'][i]*100:.2f}%")
                print(f"    Recall: {results['per_class_recall'][i]*100:.2f}%")
                print(f"    F1: {results['per_class_f1'][i]*100:.2f}%")
                print(f"    Support: {results['per_class_support'][i]}")
            except IndexError:
                print(f"  {class_name}: No data available.")
    else:
        # Evaluate on all tasks
        for task_idx, task in enumerate(tasks):
            _, val_loader = preprocessor.create_data_loaders(
                task['paths'], task['labels'],
                batch_size=config.BATCH_SIZE
            )
            
            print(f"\nEvaluating on Task {task_idx}...")
            results = evaluate_model(model, val_loader, device, class_names)
            all_results[f'task_{task_idx}'] = results
            
            print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
        
        # Overall evaluation on all data
        print("\nEvaluating on all data...")
        _, val_loader = preprocessor.create_data_loaders(
            image_paths, labels,
            batch_size=config.BATCH_SIZE
        )
        overall_results = evaluate_model(model, val_loader, device, class_names)
        all_results['overall'] = overall_results
        
        print(f"\nOverall Accuracy (All Data): {overall_results['overall_accuracy']*100:.2f}%")
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            np.array(overall_results['confusion_matrix']),
            class_names,
            cm_path
        )
        
        # Plot per-class metrics
        metrics_path = os.path.join(args.output_dir, 'per_class_metrics.png')
        plot_per_class_metrics(overall_results, class_names, metrics_path)
    
    # Save results
    suffix = f"_task{args.task_id}" if args.task_id is not None else ""
    results_path = os.path.join(args.output_dir, f'evaluation_results{suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
