"""
Demo script to test the CICL system with sample data
This script demonstrates how to use the system without requiring full datasets
"""

import torch
import numpy as np
from PIL import Image
import config
from models.cicl_model import DenseNet201CICL, ExemplarMemory, CICLTrainer


def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing"""
    img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def demo_model():
    """Demonstrate model initialization and forward pass"""
    print("="*70)
    print("CICL Model Demo")
    print("="*70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Initialize model
    print("Initializing DenseNet201 CICL model...")
    model = DenseNet201CICL(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
    model = model.to(device)
    model.eval()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Create dummy input
    print("Creating dummy input...")
    dummy_img = create_dummy_image()
    transform = config.get_augmentation_transforms if hasattr(config, 'get_augmentation_transforms') else None
    
    # Simple transform for demo
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(dummy_img).unsqueeze(0).to(device)
    print(f"Input shape: {input_tensor.shape}\n")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = output.argmax(dim=1)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {config.IDX_TO_CLASS[pred.item()]}")
    print(f"Confidence: {probs[0][pred.item()].item()*100:.2f}%\n")
    
    # Test exemplar memory
    print("Testing Exemplar Memory...")
    exemplar_memory = ExemplarMemory(memory_size=config.EXEMPLAR_MEMORY_SIZE)
    
    # Create some dummy exemplars
    dummy_images = [create_dummy_image() for _ in range(10)]
    dummy_images_tensor = torch.stack([transform(img) for img in dummy_images])
    dummy_labels = torch.tensor([0] * 10, dtype=torch.long)
    
    exemplar_memory.add_exemplars(0, dummy_images_tensor, dummy_labels, model, device)
    print(f"Added exemplars for class 0")
    
    replay_images, replay_labels = exemplar_memory.get_replay_batch(5, device)
    if replay_images is not None:
        print(f"Retrieved {len(replay_images)} replay samples\n")
    
    # Test EWC
    print("Testing EWC (Elastic Weight Consolidation)...")
    model.train()
    ewc_loss = model.ewc_loss()
    print(f"EWC loss (before computing Fisher): {ewc_loss.item()}\n")
    
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    demo_model()

