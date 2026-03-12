"""
Binary Model 3: Normal vs Abnormal Chest X-rays Classifier
=========================================================

Purpose: Pre-screen normal X-rays before CheXNet
Architecture: ResNet18 (pretrained backbone) ⭐ RTX 3050 OPTIMIZED
Parameters: ~11M
Expected accuracy: >90%
Size: ~45 MB
Speed: ~80ms on GPU

Usage:
    python binary_model3.py --train --data path/to/data
    python binary_model3.py --test --model models/normal_vs_abnormal.pth --data path/to/test_data
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import time

class NormalVsAbnormalDataset(Dataset):
    """Dataset for Normal vs Abnormal X-rays classification"""

    def __init__(self, data_dir, transform=None, is_train=True, sample_size=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []

        # Expected structure: data_dir/normal/ and data_dir/abnormal/
        normal_dir = os.path.join(data_dir, 'normal')
        abnormal_dir = os.path.join(data_dir, 'abnormal')

        normal_samples = []
        abnormal_samples = []

        if os.path.exists(normal_dir):
            for img_file in os.listdir(normal_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    normal_samples.append((os.path.join(normal_dir, img_file), 0))  # 0 = Normal

        if os.path.exists(abnormal_dir):
            for img_file in os.listdir(abnormal_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    abnormal_samples.append((os.path.join(abnormal_dir, img_file), 1))  # 1 = Abnormal

        # Apply sampling if specified
        if sample_size:
            import random
            random.seed(42)  # For reproducibility
            normal_samples = random.sample(normal_samples, min(sample_size, len(normal_samples)))
            abnormal_samples = random.sample(abnormal_samples, min(sample_size, len(abnormal_samples)))

        self.samples = normal_samples + abnormal_samples
        print(f"Loaded {len(self.samples)} samples from {data_dir} (normal: {len(normal_samples)}, abnormal: {len(abnormal_samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # handle corrupted or unreadable images gracefully
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Skipping unreadable image: {img_path} ({e})")
            # create a dummy black image so transform still works
            image = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

class NormalVsAbnormalModel(nn.Module):
    """ResNet18-based transfer learning classifier for Normal vs Abnormal X-rays
    
    Uses pretrained ResNet18 from ImageNet for better feature extraction
    and superior accuracy on medical imaging tasks. RTX 3050 optimized.
    """

    def __init__(self, num_classes=2):
        super(NormalVsAbnormalModel, self).__init__()

        # Load pretrained ResNet18 with proper weights API (RTX 3050 compatible)
        from torchvision.models import ResNet18_Weights
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace classifier head with improved architecture
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters for initial training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze only the fc layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self, unfreeze_blocks=1):
        """Progressively unfreeze backbone blocks for fine-tuning
        
        Args:
            unfreeze_blocks: Number of residual blocks to unfreeze from the end
        """
        # Unfreeze the last `unfreeze_blocks` residual blocks
        blocks = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        
        for i in range(min(unfreeze_blocks, len(blocks))):
            for param in blocks[i].parameters():
                param.requires_grad = True

def get_transforms(is_train=True):
    """Get medical imaging-specific data transforms for X-rays
    
    Includes augmentations designed for X-ray images to improve model robustness
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # Medical image augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # Slight rotation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
            # Intensity augmentations (important for X-rays)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_model(data_dir, model_path, epochs=50, batch_size=16, lr=1e-4, sample_size=15000):
    """Train the Normal vs Abnormal model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create datasets
    train_dataset = NormalVsAbnormalDataset(data_dir, transform=get_transforms(is_train=True), sample_size=sample_size)
    val_dataset = NormalVsAbnormalDataset(data_dir.replace('train', 'val') if 'train' in data_dir else data_dir,
                                      transform=get_transforms(is_train=False), sample_size=sample_size)

    if len(val_dataset) == 0:
        # If no validation split, use 80/20 split
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    # RTX 3050 optimized: smaller batch size, no multiprocessing
    safebatch_size = min(batch_size, len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=safebatch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=safebatch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = NormalVsAbnormalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with accuracy: {best_acc:.4f}')

        scheduler.step(val_acc)

    print(f'Training complete. Best validation accuracy: {best_acc:.4f}')
    return model

def _train_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch, total_epochs):
    """Removed - not needed"""
    pass

def _validate_epoch(model, val_loader, criterion, device):
    """Removed - not needed"""
    pass

def test_model(model_path, data_dir, batch_size=32):
    """Test the trained model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = NormalVsAbnormalModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create test dataset
    test_dataset = NormalVsAbnormalDataset(data_dir, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal']))

    return accuracy

def predict_image(model_path, image_path):
    """Predict single image"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = NormalVsAbnormalModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(is_train=False)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

    class_names = ['Normal', 'Abnormal']
    pred_class = class_names[preds.item()]
    confidence = probs[0][preds.item()].item()

    return pred_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normal vs Abnormal Chest X-rays Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--predict', action='store_true', help='Predict single image')
    parser.add_argument('--data', type=str, help='Path to data directory')
    parser.add_argument('--model', type=str, default='models/normal_vs_abnormal.pth', help='Path to save/load model')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50 for transfer learning)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16 for RTX 3050 4GB)')
    parser.add_argument('--sample-size', type=int, default=15000, help='Sample size per class (default: 15000 for RTX 3050 optimization)')

    args = parser.parse_args()

    if args.train:
        if not args.data:
            print("Please provide --data path for training")
            exit(1)
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        train_model(args.data, args.model, args.epochs, args.batch_size, sample_size=args.sample_size)
    elif args.test:
        if not args.model or not args.data:
            print("Please provide --model and --data paths for testing")
            exit(1)
        test_model(args.model, args.data)
    elif args.predict:
        if not args.model or not args.image:
            print("Please provide --model and --image paths for prediction")
            exit(1)
        pred_class, confidence = predict_image(args.model, args.image)
        print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")
    else:
        print("Please specify --train, --test, or --predict")
