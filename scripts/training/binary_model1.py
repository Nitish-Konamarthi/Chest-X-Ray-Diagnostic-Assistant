"""
Binary Model 1: Garbage vs X-ray Classifier
===========================================

Purpose: Filter out photos, CT scans, MRI, garbage images from actual X-rays
Architecture: MobileNetV2 (pretrained backbone)
Parameters: ~3.5M
Expected accuracy: 93-95%
Size: ~14 MB
Speed: ~50ms on GPU

Usage:
    python binary_model1.py --train --data path/to/data
    python binary_model1.py --test --model models/garbage_vs_xray.pth --data path/to/test_data
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

class GarbageVsXrayDataset(Dataset):
    """Dataset for Garbage vs X-ray classification"""

    def __init__(self, data_dir, transform=None, is_train=True, sample_size=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []

        # Expected structure: data_dir/xray/ and data_dir/garbage/
        xray_dir = os.path.join(data_dir, 'xray')
        garbage_dir = os.path.join(data_dir, 'garbage')

        xray_samples = []
        garbage_samples = []

        if os.path.exists(xray_dir):
            for img_file in os.listdir(xray_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    xray_samples.append((os.path.join(xray_dir, img_file), 1))  # 1 = X-ray

        if os.path.exists(garbage_dir):
            for img_file in os.listdir(garbage_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    garbage_samples.append((os.path.join(garbage_dir, img_file), 0))  # 0 = Garbage

        # Apply sampling if specified
        if sample_size:
            import random
            random.seed(42)  # For reproducibility
            xray_samples = random.sample(xray_samples, min(sample_size, len(xray_samples)))
            garbage_samples = random.sample(garbage_samples, min(sample_size, len(garbage_samples)))

        self.samples = xray_samples + garbage_samples
        print(f"Loaded {len(self.samples)} samples from {data_dir} (xray: {len(xray_samples)}, garbage: {len(garbage_samples)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class GarbageVsXrayModel(nn.Module):
    """MobileNetV2-based binary classifier for Garbage vs X-ray"""

    def __init__(self, num_classes=2):
        super(GarbageVsXrayModel, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_transforms(is_train=True):
    """Get data transforms for training/validation"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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

def train_model(data_dir, model_path, epochs=20, batch_size=32, lr=1e-4, sample_size=None):
    """Train the Garbage vs X-ray model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create datasets
    train_dataset = GarbageVsXrayDataset(data_dir, transform=get_transforms(is_train=True), sample_size=sample_size)
    val_dataset = GarbageVsXrayDataset(data_dir.replace('train', 'val') if 'train' in data_dir else data_dir,
                                       transform=get_transforms(is_train=False), sample_size=sample_size)

    if len(val_dataset) == 0:
        # If no validation split, use 80/20 split
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = GarbageVsXrayModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with accuracy: {best_acc:.4f}')

        scheduler.step()

    print(f'Training complete. Best validation accuracy: {best_acc:.4f}')
    return model

def test_model(model_path, data_dir, batch_size=32):
    """Test the trained model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = GarbageVsXrayModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create test dataset
    test_dataset = GarbageVsXrayDataset(data_dir, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
    print(classification_report(all_labels, all_preds, target_names=['Garbage', 'X-ray']))

    return accuracy

def predict_image(model_path, image_path):
    """Predict single image"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = GarbageVsXrayModel()
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

    class_names = ['Garbage', 'X-ray']
    pred_class = class_names[preds.item()]
    confidence = probs[0][preds.item()].item()

    return pred_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Garbage vs X-ray Classifier')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--predict', action='store_true', help='Predict single image')
    parser.add_argument('--data', type=str, help='Path to data directory')
    parser.add_argument('--model', type=str, default='models/garbage_vs_xray.pth', help='Path to save/load model')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size per class (None = use all data)')

    args = parser.parse_args()

    if args.train:
        if not args.data:
            print("Please provide --data path for training")
            exit(1)
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        train_model(args.data, args.model, args.epochs, args.batch_size, sample_size=args.sample_size)

    elif args.test:
        if not args.data:
            print("Please provide --data path for testing")
            exit(1)
        test_model(args.model, args.data, args.batch_size)

    elif args.predict:
        if not args.image:
            print("Please provide --image path for prediction")
            exit(1)
        pred_class, confidence = predict_image(args.model, args.image)
        print(f"Prediction: {pred_class} (Confidence: {confidence:.4f})")

    else:
        print("Please specify --train, --test, or --predict")