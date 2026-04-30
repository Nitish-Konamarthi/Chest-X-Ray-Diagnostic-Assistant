"""
ChestXRayAnalyzer — high-level Python SDK for chest X-ray analysis.

This is the main public interface for developers who ``pip install chest-xray-diagnosis``
and want to use the models programmatically without running a server.

Usage::

    from chexray_diagnosis import ChestXRayAnalyzer

    analyzer = ChestXRayAnalyzer()          # auto-downloads models on first run
    result = analyzer.analyze("xray.png")

    print(result["is_normal"])
    print(result["pathologies"])
    print(result["heatmap_b64"])
    print(result["ai_explanation"])
"""

import io
import base64
import collections
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from chexray_diagnosis.config import (
    CLASS_NAMES,
    CLINICAL_THRESHOLDS,
    get_gemini_api_key,
    get_geoapify_api_key,
)
from chexray_diagnosis.models.download import (
    download_models,
    get_model_path,
    models_present,
)
from chexray_diagnosis.models.densenet import DenseNet121
from chexray_diagnosis.models.binary_pipeline import BinaryClassifierPipeline
from chexray_diagnosis.services.heatmap import generate_gradcam
from chexray_diagnosis.services.explainability import ExplainabilityAI
from chexray_diagnosis.services.geoapify import GeoapifyDoctorFinder


class ChestXRayAnalyzer:
    """End-to-end chest X-ray analysis pipeline.

    Args:
        gemini_api_key:  Explicit Gemini key (or set ``GEMINI_API_KEY`` env var).
        geoapify_api_key: Explicit Geoapify key (or set ``GEOAPIFY_API_KEY`` env var).
        device:          ``"cuda"`` or ``"cpu"`` (auto-detected if ``None``).
        auto_download:   Download model weights if missing (default ``True``).
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        geoapify_api_key: Optional[str] = None,
        device: Optional[str] = None,
        auto_download: bool = True,
    ):
        # Ensure models exist
        if not models_present():
            if auto_download:
                print("📥 Models not found — downloading from HuggingFace Hub...")
                download_models()
            else:
                raise FileNotFoundError(
                    "Model files not found. Run `chest-xray-diagnosis download-models` first."
                )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"🔧 Device: {self.device}")

        # Load CheXNet
        self.model = self._load_chexnet()

        # Load binary pipeline
        self.binary_pipeline = BinaryClassifierPipeline(
            model1_path=str(get_model_path("binary", "garbage_vs_xray")),
            model2_path=str(get_model_path("binary", "chest_vs_other")),
            model3_path=str(get_model_path("binary", "normal_vs_abnormal")),
        )

        # Services (initialised lazily with provided or env-var keys)
        self.explainability = ExplainabilityAI(api_key=gemini_api_key)
        self.doctor_finder = GeoapifyDoctorFinder(api_key=geoapify_api_key)

        # Image preprocessing
        self.preprocess = T.Compose([
            T.Resize(256),
            T.TenCrop(224),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(c) for c in crops])),
            T.Lambda(lambda crops: torch.stack([
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(c)
                for c in crops
            ])),
        ])

        print("✅ ChestXRayAnalyzer ready")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_chexnet(self) -> DenseNet121:
        path = get_model_path("chexnet", "main")
        ckpt = torch.load(str(path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        cleaned = collections.OrderedDict(
            (k.replace("module.", ""), v) for k, v in state.items()
        )
        model = DenseNet121(14, True)
        model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()
        print("✅ CheXNet model loaded")
        return model

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_chexnet(
        self, image_gray: Image.Image
    ) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
        arr = np.stack([np.array(image_gray)] * 3, axis=-1).astype(np.uint8)
        inp = self.preprocess(Image.fromarray(arr)).to(self.device)

        with torch.no_grad():
            out = self.model(inp)
            probs = torch.sigmoid(out).mean(dim=0)

        pairs = sorted(
            zip(CLASS_NAMES, probs.cpu().numpy().tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs, inp[0].unsqueeze(0)

    @staticmethod
    def _classify_confidence(prob: float) -> str:
        if prob >= 0.65:
            return "HIGH"
        if prob >= 0.40:
            return "MEDIUM"
        if prob >= 0.20:
            return "LOW"
        return "MINIMAL"

    @staticmethod
    def _clinical_summary(
        results: List[Tuple[str, float]], model3_normal: bool
    ) -> Tuple[str, str]:
        high = [(n, p) for n, p in results if p >= 0.65]
        med = [(n, p) for n, p in results if 0.40 <= p < 0.65]

        if model3_normal and not high and not med:
            return "✅ NORMAL — No significant abnormalities detected", "NORMAL"
        if model3_normal and med:
            names = ", ".join(n for n, _ in med)
            return f"⚠️ LOW CONFIDENCE — Minor findings: {names}", "BORDERLINE"
        if not model3_normal or high:
            names = ", ".join(n for n, _ in (high or med))
            return f"⚠️ ABNORMAL — Significant findings: {names}", "ABNORMAL"
        return "Analysis completed — review detailed results below", "UNKNOWN"

    def _heatmap_b64(
        self, image_rgb: Image.Image, inp: torch.Tensor
    ) -> Optional[str]:
        try:
            heatmap_np = generate_gradcam(self.model, inp, self.device, image_rgb, 0.5)
            if heatmap_np is None:
                return None
            pil = Image.fromarray(heatmap_np)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Analyse a chest X-ray image end-to-end.

        Args:
            image_path: Path to a PNG/JPG/TIFF X-ray image.

        Returns:
            Dictionary with keys:
                ``binary_pipeline``, ``valid_for_analysis``, ``is_normal``,
                ``clinical_summary``, ``assessment_level``, ``pathologies``,
                ``heatmap_b64``, ``ai_explanation``, ``api_used``,
                ``processing_time_ms``.
        """
        start = time.time()

        image_gray = Image.open(image_path).convert("L")
        image_rgb = Image.open(image_path).convert("RGB")

        # Stage 1 — binary pipeline
        validation = self.binary_pipeline.validate(image_rgb)

        binary_results = []
        r = validation["results"] or {}
        m1 = r.get("model1")
        m2 = r.get("model2")
        m3 = r.get("model3")

        binary_results.append({
            "model": "Image Validation",
            "is_valid": m1.get("is_xray", False) if m1 else False,
            "confidence": m1.get("confidence", 0.0) if m1 else 0.0,
            "message": "Valid X-ray image" if (m1 and m1.get("is_xray")) else "Invalid image type",
        })
        binary_results.append({
            "model": "Anatomy Check",
            "is_valid": m2.get("is_chest", False) if m2 else False,
            "confidence": m2.get("confidence", 0.0) if m2 else 0.0,
            "message": "Chest X-ray confirmed" if (m2 and m2.get("is_chest")) else "Not a chest X-ray",
        })

        is_normal = m3.get("prediction", "").lower() == "normal" if m3 else False
        binary_results.append({
            "model": "Initial Assessment",
            "is_valid": True,
            "confidence": m3.get("confidence", 0.0) if m3 else 0.0,
            "message": "Normal findings" if is_normal else "Abnormal findings detected",
        })

        if not validation["valid"]:
            return {
                "binary_pipeline": binary_results,
                "valid_for_analysis": False,
                "is_normal": False,
                "clinical_summary": validation["message"],
                "assessment_level": "INVALID",
                "pathologies": None,
                "heatmap_b64": None,
                "ai_explanation": "Image validation failed. Please upload a valid chest X-ray.",
                "api_used": "System",
                "processing_time_ms": (time.time() - start) * 1000,
            }

        # Stage 2 — NORMAL path
        if validation.get("skip_main_model", False) or is_normal:
            ai = self.explainability.generate_normal_explanation(m3 or {})
            return {
                "binary_pipeline": binary_results,
                "valid_for_analysis": True,
                "is_normal": True,
                "clinical_summary": "✅ NORMAL — No significant abnormalities detected",
                "assessment_level": "NORMAL",
                "pathologies": None,
                "heatmap_b64": None,
                "ai_explanation": ai["explanation"],
                "api_used": ai["api_used"],
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            }

        # Stage 3 — ABNORMAL path (full CheXNet)
        results, inp = self._predict_chexnet(image_gray)
        summary_text, assessment_level = self._clinical_summary(results, False)

        pathologies = []
        for disease, prob in results:
            is_positive = prob >= CLINICAL_THRESHOLDS.get(disease, 0.5)
            pathologies.append({
                "pathology": disease,
                "probability": round(prob, 4),
                "confidence_level": self._classify_confidence(prob),
                "clinical_decision": "POSITIVE" if is_positive else "NEGATIVE",
            })

        heatmap = self._heatmap_b64(image_rgb, inp)

        main_result = {
            "assessment_level": assessment_level,
            "pathologies": [
                {"name": p["pathology"], "probability": p["probability"]}
                for p in pathologies
            ],
        }
        ai = self.explainability.generate_abnormal_explanation(m3 or {}, main_result)

        return {
            "binary_pipeline": binary_results,
            "valid_for_analysis": True,
            "is_normal": False,
            "clinical_summary": summary_text,
            "assessment_level": assessment_level,
            "pathologies": pathologies,
            "heatmap_b64": heatmap,
            "ai_explanation": ai["explanation"],
            "api_used": ai["api_used"],
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }
