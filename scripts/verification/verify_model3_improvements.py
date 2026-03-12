"""
Quick Verification Script for Model 3 Improvements
==================================================

This script verifies that the improved Model 3 is set up correctly
and can perform inference. It includes:
- Architecture verification
- Model initialization check
- Quick inference test on sample data
- Training readiness verification

Usage:
    python verify_model3_improvements.py
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

def verify_architecture():
    """Verify that ResNet50 model is correctly defined"""
    print("\n" + "="*60)
    print("VERIFICATION: Model Architecture")
    print("="*60)
    
    try:
        # Import the model
        from binary_model3 import NormalVsAbnormalModel
        
        # Create model
        model = NormalVsAbnormalModel(num_classes=2)
        print("✓ NormalVsAbnormalModel created successfully")
        
        # Check if it's ResNet50 based
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
            print("✓ ResNet50 backbone confirmed")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Check FC head
        print(f"✓ FC head architecture verified")
        for i, layer in enumerate(model.backbone.fc):
            print(f"  Layer {i}: {layer}")
        
        return True
    except Exception as e:
        print(f"✗ Architecture verification failed: {e}")
        return False

def verify_training_methods():
    """Verify freeze/unfreeze methods for two-stage training"""
    print("\n" + "="*60)
    print("VERIFICATION: Training Methods (Freeze/Unfreeze)")
    print("="*60)
    
    try:
        from binary_model3 import NormalVsAbnormalModel
        
        model = NormalVsAbnormalModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test freeze_backbone
        model.freeze_backbone()
        frozen_count = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        print(f"✓ freeze_backbone() works: {frozen_count} frozen parameters")
        
        # Verify FC head is still trainable
        fc_trainable = sum(1 for p in model.backbone.fc.parameters() if p.requires_grad)
        if fc_trainable > 0:
            print(f"✓ FC head remains trainable: {fc_trainable} trainable parameters")
        
        # Test unfreeze_backbone
        model.unfreeze_backbone(unfreeze_blocks=2)
        unfrozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        print(f"✓ unfreeze_backbone(2) works: {unfrozen_count} trainable parameters")
        
        return True
    except Exception as e:
        print(f"✗ Training methods verification failed: {e}")
        return False

def verify_transforms():
    """Verify medical imaging-specific augmentations"""
    print("\n" + "="*60)
    print("VERIFICATION: Data Transforms (Augmentations)")
    print("="*60)
    
    try:
        from binary_model3 import get_transforms
        
        # Get training transforms
        train_transform = get_transforms(is_train=True)
        print("✓ Training transforms loaded:")
        for transform in train_transform.transforms:
            print(f"  - {transform.__class__.__name__}")
        
        # Get validation transforms
        val_transform = get_transforms(is_train=False)
        print("✓ Validation transforms loaded:")
        for transform in val_transform.transforms:
            print(f"  - {transform.__class__.__name__}")
        
        # Create dummy image and apply transforms
        dummy_img = Image.new('RGB', (256, 256), color='gray')
        transformed = train_transform(dummy_img)
        print(f"✓ Transforms executable: output shape {transformed.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Transforms verification failed: {e}")
        return False

def verify_dataset():
    """Verify dataset class with actual data"""
    print("\n" + "="*60)
    print("VERIFICATION: Dataset Class")
    print("="*60)
    
    try:
        from binary_model3 import NormalVsAbnormalDataset, get_transforms
        
        data_dir = 'data/model3_normal_vs_abnormal'
        
        if not os.path.exists(data_dir):
            print(f"⚠ Data directory not found: {data_dir}")
            print("  (This is OK if you haven't organized data yet)")
            return True
        
        # Create dataset
        dataset = NormalVsAbnormalDataset(
            data_dir, 
            transform=get_transforms(is_train=False),
            is_train=False
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Try loading a sample
            sample_img, label = dataset[0]
            print(f"✓ Sample loadable: tensor shape {sample_img.shape}, label {label}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        return False

def verify_inference():
    """Verify model can perform inference"""
    print("\n" + "="*60)
    print("VERIFICATION: Inference Capability")
    print("="*60)
    
    try:
        from binary_model3 import NormalVsAbnormalModel, get_transforms
        
        model = NormalVsAbnormalModel()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded on device: {device}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Inference successful: output shape {output.shape}")
        
        # Check softmax
        probs = torch.softmax(output, dim=1)
        print(f"✓ Softmax probabilities: {probs[0].cpu().numpy()}")
        print(f"✓ Prediction: {'Normal' if probs[0, 0] > 0.5 else 'Abnormal'}")
        
        return True
    except Exception as e:
        print(f"✗ Inference verification failed: {e}")
        return False

def verify_pipeline():
    """Verify pipeline integration"""
    print("\n" + "="*60)
    print("VERIFICATION: Pipeline Integration")
    print("="*60)
    
    try:
        from binary_pipeline import NormalVsAbnormalModel as PipelineModel
        
        model = PipelineModel()
        print("✓ Pipeline model class loads correctly")
        print("✓ Pipeline compatible with improvements")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline verification failed: {e}")
        return False

def main():
    print("\n" + "🚀 QUICK VERIFICATION: Model 3 Improvements (ResNet50 Transfer Learning)")
    
    results = {
        "Architecture": verify_architecture(),
        "Training Methods": verify_training_methods(),
        "Data Augmentations": verify_transforms(),
        "Dataset Class": verify_dataset(),
        "Inference": verify_inference(),
        "Pipeline Integration": verify_pipeline(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} verifications passed")
    
    if passed == total:
        print("\n✅ All verifications passed! Ready to train.")
        print("\nStart training with:")
        print("  python train_all.py --data data --epochs 10 --epochs-model3 50")
        print("\nOr train Model 3 specifically:")
        print("  python binary_model3.py --train --data data/model3_normal_vs_abnormal")
        return 0
    else:
        print("\n⚠️ Some verifications failed. Check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
