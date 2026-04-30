"""
Binary Classification Pipeline
================================

Three-stage validation pipeline for chest X-ray images:
    Model 1: Garbage vs. X-ray (MobileNetV2)
    Model 2: Chest vs. Other X-ray (ResNet18)
    Model 3: Normal vs. Abnormal (ResNet18)

Usage::

    from chexray_diagnosis.models.binary_pipeline import BinaryClassifierPipeline

    pipeline = BinaryClassifierPipeline(model1_path, model2_path, model3_path)
    result = pipeline.validate(pil_image)
"""

import os
import time
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class GarbageVsXrayModel(nn.Module):
    """MobileNetV2-based binary classifier: Garbage vs X-ray."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, TypeError):
            self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ChestVsOtherModel(nn.Module):
    """ResNet18-based binary classifier: Chest vs Other X-rays."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, TypeError):
            self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class NormalVsAbnormalModel(nn.Module):
    """ResNet18-based classifier: Normal vs Abnormal chest X-rays."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
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
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BinaryClassifierPipeline:
    """
    Three-model validation pipeline for chest X-ray images.

    Decision flow:
        * Model 1 rejects non-X-ray images.
        * Model 2 rejects non-chest X-rays.
        * Model 3 classifies Normal vs Abnormal.
            - Normal  → ``skip_main_model=True``  (health tips)
            - Abnormal → ``proceed_to_main_model=True`` (full CheXNet)
    """

    def __init__(
        self,
        model1_path: str,
        model2_path: str,
        model3_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate & load
        self.model1 = self._load(GarbageVsXrayModel(), model1_path)
        self.model2 = self._load(ChestVsOtherModel(), model2_path)
        self.model3 = self._load(NormalVsAbnormalModel(), model3_path)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ---- helpers ----------------------------------------------------------

    def _load(self, model: nn.Module, path: str) -> nn.Module:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        model = model.to(self.device)
        model.load_state_dict(
            torch.load(path, map_location=self.device), strict=False,
        )
        model.eval()
        return model

    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or file path string")
        return self.transform(image).unsqueeze(0).to(self.device)

    @staticmethod
    def _binary_predict(
        model: nn.Module, tensor: torch.Tensor
    ) -> Tuple[bool, float]:
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidence = probs[0][preds.item()].item()
            is_positive = preds.item() == 1
        return is_positive, confidence

    # ---- public API -------------------------------------------------------

    def validate(self, image: Union[str, Image.Image], threshold: float = 0.7) -> Dict[str, Any]:
        """Run the 3-stage validation pipeline.

        Returns a dict with keys: ``valid``, ``message``, ``is_normal``,
        ``proceed_to_main_model``, ``skip_main_model``, ``results``.
        """
        start = time.time()

        try:
            tensor = self._preprocess(image)

            # Stage 1 — Garbage vs X-ray
            is_xray, xray_conf = self._binary_predict(self.model1, tensor)
            if not is_xray or xray_conf < threshold:
                return self._result(
                    valid=False,
                    message="❌ This doesn't appear to be an X-ray image. Please upload a medical X-ray.",
                    results={
                        "model1": {"is_xray": is_xray, "confidence": xray_conf},
                        "model2": None,
                        "model3": None,
                    },
                )

            # Stage 2 — Chest vs Other
            is_chest, chest_conf = self._binary_predict(self.model2, tensor)
            if not is_chest or chest_conf < threshold:
                return self._result(
                    valid=False,
                    message="❌ This appears to be an X-ray, but not of the chest.",
                    results={
                        "model1": {"is_xray": is_xray, "confidence": xray_conf},
                        "model2": {"is_chest": is_chest, "confidence": chest_conf},
                        "model3": None,
                    },
                )

            # Stage 3 — Normal vs Abnormal
            with torch.no_grad():
                outputs = self.model3(tensor)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                pred_idx = preds.item()
                abnormal_conf = probs[0][pred_idx].item()
                is_abnormal = pred_idx == 1
                prediction = "abnormal" if is_abnormal else "normal"

            elapsed_ms = (time.time() - start) * 1000
            is_normal = not is_abnormal

            results = {
                "model1": {"is_xray": is_xray, "confidence": xray_conf},
                "model2": {"is_chest": is_chest, "confidence": chest_conf},
                "model3": {
                    "prediction": prediction,
                    "is_abnormal": is_abnormal,
                    "confidence": abnormal_conf,
                },
                "processing_time_ms": elapsed_ms,
            }

            if is_normal:
                return self._result(
                    valid=True,
                    message="✅ Valid chest X-ray detected. Initial assessment: NORMAL.",
                    is_normal=True,
                    proceed=False,
                    skip=True,
                    results=results,
                )
            else:
                return self._result(
                    valid=True,
                    message="✅ Valid chest X-ray detected. Initial assessment: ABNORMAL — proceeding to detailed analysis.",
                    is_normal=False,
                    proceed=True,
                    skip=False,
                    results=results,
                )

        except Exception as e:
            return self._result(
                valid=False,
                message=f"❌ Error processing image: {e}",
            )

    @staticmethod
    def _result(
        valid: bool = False,
        message: str = "",
        is_normal: bool = False,
        proceed: bool = False,
        skip: bool = False,
        results: Any = None,
    ) -> Dict[str, Any]:
        return {
            "valid": valid,
            "message": message,
            "is_normal": is_normal,
            "proceed_to_main_model": proceed,
            "skip_main_model": skip,
            "results": results,
        }

    def predict_all(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        """Get raw predictions from all 3 models without validation logic."""
        tensor = self._preprocess(image)
        is_xray, xray_conf = self._binary_predict(self.model1, tensor)
        is_chest, chest_conf = self._binary_predict(self.model2, tensor)

        with torch.no_grad():
            outputs = self.model3(tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            pred_idx = preds.item()
            abnormal_conf = probs[0][pred_idx].item()
            is_abnormal = pred_idx == 1
            prediction = "abnormal" if is_abnormal else "normal"

        return {
            "model1_garbage_vs_xray": {
                "prediction": "X-ray" if is_xray else "Garbage",
                "confidence": xray_conf,
                "is_xray": is_xray,
            },
            "model2_chest_vs_other": {
                "prediction": "Chest X-ray" if is_chest else "Other X-ray",
                "confidence": chest_conf,
                "is_chest": is_chest,
            },
            "model3_normal_vs_abnormal": {
                "prediction": prediction,
                "confidence": abnormal_conf,
                "is_abnormal": is_abnormal,
            },
        }
