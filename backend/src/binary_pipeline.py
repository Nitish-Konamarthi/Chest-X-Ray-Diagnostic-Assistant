"""
Binary Classification Pipeline
=============================

ENHANCED VERSION with modified decision flow:
- Model3 (Normal) → Skip CheXNet, show health tips  
- Model3 (Abnormal) → Proceed to full CheXNet analysis

Usage in app.py:
    from binary_pipeline import BinaryClassifierPipeline

    pipeline = BinaryClassifierPipeline(
        model1='models/garbage_vs_xray.pth',
        model2='models/chest_vs_other.pth',
        model3='models/normal_vs_abnormal.pth'
    )

    result = pipeline.validate(image)
    if not result['valid']:
        st.error(result['message'])
        st.stop()
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

class GarbageVsXrayModel(nn.Module):
    """MobileNetV2-based binary classifier for Garbage vs X-ray"""
    def __init__(self, num_classes=2):
        super(GarbageVsXrayModel, self).__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, TypeError):
            self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class ChestVsOtherModel(nn.Module):
    """ResNet18-based binary classifier for Chest vs Other X-rays.

    NOTE: Uses ResNet18 (not ResNet50) — the chest_vs_other.pth weights were
    trained on ResNet18 (3x3 conv layers, 512-dim fc features).
    """
    def __init__(self, num_classes=2):
        super(ChestVsOtherModel, self).__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, TypeError):
            self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

class NormalVsAbnormalModel(nn.Module):
    """ResNet18-based transfer learning classifier for Normal vs Abnormal X-rays
    
    Updated with better architecture for improved accuracy (>90% target)
    """
    def __init__(self, num_classes=2):
        super(NormalVsAbnormalModel, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        # Replace classifier head
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

class BinaryClassifierPipeline:
    """
    Pipeline combining 3 binary classifiers for X-ray validation

    ENHANCED: Modified decision flow
    - Normal → Skip CheXNet (proceed_to_main_model=False, skip_main_model=True)
    - Abnormal → Run CheXNet (proceed_to_main_model=True, skip_main_model=False)

    Returns validation results with clear error messages for seamless UI integration
    """

    def __init__(self, model1_path, model2_path, model3_path):
        """
        Initialize pipeline with 3 model paths

        Args:
            model1_path: Path to garbage_vs_xray.pth
            model2_path: Path to chest_vs_other.pth
            model3_path: Path to normal_vs_abnormal.pth
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Binary Pipeline using device: {self.device}")

        # Initialize models
        self.model1 = GarbageVsXrayModel().to(self.device)
        self.model2 = ChestVsOtherModel().to(self.device)
        self.model3 = NormalVsAbnormalModel().to(self.device)

        # Load model weights
        if os.path.exists(model1_path):
            self.model1.load_state_dict(torch.load(model1_path, map_location=self.device), strict=False)
            self.model1.eval()
            print(f"✅ Loaded Model 1: {model1_path}")
        else:
            raise FileNotFoundError(f"Model 1 not found: {model1_path}")

        if os.path.exists(model2_path):
            self.model2.load_state_dict(torch.load(model2_path, map_location=self.device), strict=False)
            self.model2.eval()
            print(f"✅ Loaded Model 2: {model2_path}")
        else:
            raise FileNotFoundError(f"Model 2 not found: {model2_path}")

        if os.path.exists(model3_path):
            self.model3.load_state_dict(torch.load(model3_path, map_location=self.device), strict=False)
            self.model3.eval()
            print(f"✅ Loaded Model 3: {model3_path}")
        else:
            raise FileNotFoundError(f"Model 3 not found: {model3_path}")

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _preprocess_image(self, image):
        """Preprocess PIL image for all models"""
        if isinstance(image, str):
            # If path provided, load image
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image or path string")

        return self.transform(image).unsqueeze(0).to(self.device)

    def _predict_model1(self, image_tensor):
        """Model 1: Garbage vs X-ray"""
        with torch.no_grad():
            outputs = self.model1(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidence = probs[0][preds.item()].item()
            is_xray = preds.item() == 1  # 1 = X-ray, 0 = Garbage
            return is_xray, confidence

    def _predict_model2(self, image_tensor):
        """Model 2: Chest vs Other X-rays"""
        with torch.no_grad():
            outputs = self.model2(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidence = probs[0][preds.item()].item()
            is_chest = preds.item() == 1  # 1 = Chest, 0 = Other
            return is_chest, confidence

    def _predict_model3(self, image_tensor):
        """Model 3: Normal vs Abnormal — runs actual model inference."""
        with torch.no_grad():
            outputs = self.model3(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            pred_idx = preds.item()
            confidence = probs[0][pred_idx].item()
            # Class 0 = Normal, Class 1 = Abnormal
            is_abnormal = (pred_idx == 1)
            prediction = 'abnormal' if is_abnormal else 'normal'
            return is_abnormal, confidence, prediction

    def validate(self, image, threshold=0.7):
        """
        Validate image through all 3 binary classifiers

        Args:
            image: PIL Image or path string
            threshold: Confidence threshold for decisions

        Returns:
            dict: {
                'valid': bool,                  # Whether image passes all checks
                'message': str,                 # Error message if invalid
                'is_normal': bool,              # Whether image is normal (for UI hints)
                'proceed_to_main_model': bool,  # ADDED: Whether to run CheXNet
                'skip_main_model': bool,        # ADDED: Whether to skip CheXNet (normal case)
                'results': dict                 # Detailed results from each model
            }
        """

        start_time = time.time()

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)

            # Model 1: Garbage vs X-ray
            is_xray, xray_conf = self._predict_model1(image_tensor)

            if not is_xray or xray_conf < threshold:
                return {
                    'valid': False,
                    'message': "❌ This doesn't appear to be an X-ray image. Please upload a medical X-ray.",
                    'is_normal': False,
                    'proceed_to_main_model': False,  # ADDED
                    'skip_main_model': False,         # ADDED
                    'results': {
                        'model1': {'is_xray': is_xray, 'confidence': xray_conf},
                        'model2': None,
                        'model3': None
                    }
                }

            # Model 2: Chest vs Other X-rays
            is_chest, chest_conf = self._predict_model2(image_tensor)

            if not is_chest or chest_conf < threshold:
                return {
                    'valid': False,
                    'message': "❌ This appears to be an X-ray, but not of the chest. Please upload a chest X-ray.",
                    'is_normal': False,
                    'proceed_to_main_model': False,  # ADDED
                    'skip_main_model': False,         # ADDED
                    'results': {
                        'model1': {'is_xray': is_xray, 'confidence': xray_conf},
                        'model2': {'is_chest': is_chest, 'confidence': chest_conf},
                        'model3': None
                    }
                }

            # Model 3: Normal vs Abnormal — CRITICAL DECISION POINT
            is_abnormal, abnormal_conf, prediction3 = self._predict_model3(image_tensor)

            processing_time = time.time() - start_time
            is_normal = not is_abnormal

            # MODIFIED DECISION LOGIC:
            # If NORMAL → Skip main model, show health tips
            # If ABNORMAL → Proceed to main model for detailed analysis
            
            if is_normal:
                return {
                    'valid': True,
                    'message': "✅ Valid chest X-ray detected. Initial assessment: NORMAL - No significant abnormalities detected.",
                    'is_normal': True,
                    'proceed_to_main_model': False,  # ADDED: Skip CheXNet for normal cases
                    'skip_main_model': True,          # ADDED: Flag to show health tips instead
                    'results': {
                        'model1': {'is_xray': is_xray, 'confidence': xray_conf},
                        'model2': {'is_chest': is_chest, 'confidence': chest_conf},
                        'model3': {
                            'prediction': prediction3,      # 'normal'
                            'is_abnormal': is_abnormal,     # False
                            'confidence': abnormal_conf
                        },
                        'processing_time_ms': processing_time * 1000
                    }
                }
            else:
                return {
                    'valid': True,
                    'message': "✅ Valid chest X-ray detected. Initial assessment: ABNORMAL - Proceeding to detailed pathology analysis.",
                    'is_normal': False,
                    'proceed_to_main_model': True,   # ADDED: Proceed to CheXNet for abnormal cases
                    'skip_main_model': False,         # ADDED: Flag for full analysis
                    'results': {
                        'model1': {'is_xray': is_xray, 'confidence': xray_conf},
                        'model2': {'is_chest': is_chest, 'confidence': chest_conf},
                        'model3': {
                            'prediction': prediction3,      # 'abnormal'
                            'is_abnormal': is_abnormal,     # True
                            'confidence': abnormal_conf
                        },
                        'processing_time_ms': processing_time * 1000
                    }
                }

        except Exception as e:
            return {
                'valid': False,
                'message': f"❌ Error processing image: {str(e)}",
                'is_normal': False,
                'proceed_to_main_model': False,  # ADDED
                'skip_main_model': False,         # ADDED
                'results': None
            }

    def predict_all(self, image):
        """
        Get predictions from all 3 models without validation logic

        Args:
            image: PIL Image or path string

        Returns:
            dict: Raw predictions from each model
        """
        image_tensor = self._preprocess_image(image)

        is_xray, xray_conf = self._predict_model1(image_tensor)
        is_chest, chest_conf = self._predict_model2(image_tensor)
        is_abnormal, abnormal_conf, prediction3 = self._predict_model3(image_tensor)

        return {
            'model1_garbage_vs_xray': {
                'prediction': 'X-ray' if is_xray else 'Garbage',
                'confidence': xray_conf,
                'is_xray': is_xray
            },
            'model2_chest_vs_other': {
                'prediction': 'Chest X-ray' if is_chest else 'Other X-ray',
                'confidence': chest_conf,
                'is_chest': is_chest
            },
            'model3_normal_vs_abnormal': {
                'prediction': prediction3,   # 'normal' or 'abnormal'
                'confidence': abnormal_conf,
                'is_abnormal': is_abnormal
            }
        }

# Convenience function for easy integration
def create_pipeline(model_dir='models'):
    """Create pipeline with default model paths"""
    return BinaryClassifierPipeline(
        model1_path=os.path.join(model_dir, 'garbage_vs_xray.pth'),
        model2_path=os.path.join(model_dir, 'chest_vs_other.pth'),
        model3_path=os.path.join(model_dir, 'normal_vs_abnormal.pth')
    )
