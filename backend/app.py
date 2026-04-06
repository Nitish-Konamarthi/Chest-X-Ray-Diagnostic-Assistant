"""
FastAPI Backend for Chest X-ray Analysis
=========================================

ENHANCED VERSION with:
- Modified decision flow: Normal → Health tips; Abnormal → CheXNet analysis
- Geoapify integration for finding nearby specialized doctors
- Gemini 2.0 Flash for AI explanations

Endpoints:
- POST /analyze: Upload X-ray image and get analysis results
- POST /find-doctors: Find nearby doctors based on location and pathology
- GET /health: Health check

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import io
import os
import base64
import collections
from typing import Dict, List, Tuple, Optional, Any
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# Load .env from project root (backend/ is one level below project root)
_dotenv_path = find_dotenv(usecwd=False)
if _dotenv_path:
    load_dotenv(_dotenv_path)
else:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _env = os.path.join(_root, '.env')
    if os.path.exists(_env):
        load_dotenv(_env)

# Add current directory and parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models and utilities
try:
    from src.DensenetModels import DenseNet121
    from src.HeatmapGenerator import generate_gradcam
    from src.binary_pipeline import BinaryClassifierPipeline
    from src.explainability_ai import explainability_ai
    from src.geoapify_service import geoapify_finder  # NEW IMPORT
except Exception as e:
    print(f"Import error: {e}")
    raise

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "chexnet", "m-30012020-104001.pth.tar")
CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
               'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
               'Fibrosis','Pleural_Thickening','Hernia']
CLINICAL_THRESHOLDS = {
    'Atelectasis': 0.45, 'Cardiomegaly': 0.55, 'Effusion': 0.50, 'Infiltration': 0.35,
    'Mass': 0.60, 'Nodule': 0.55, 'Pneumonia': 0.40, 'Pneumothorax': 0.50,
    'Consolidation': 0.45, 'Edema': 0.55, 'Emphysema': 0.60, 'Fibrosis': 0.55,
    'Pleural_Thickening': 0.45, 'Hernia': 0.70
}

# ============================================================================
# Pydantic models for responses
# ============================================================================

class BinaryResult(BaseModel):
    model: str
    is_valid: bool
    confidence: float
    message: str

class PathologyResult(BaseModel):
    pathology: str
    probability: float
    confidence_level: str
    clinical_decision: str

# MODIFIED: Added is_normal field, made pathologies optional
class AnalysisResponse(BaseModel):
    binary_pipeline: List[BinaryResult]
    valid_for_analysis: bool
    is_normal: bool  # NEW: Critical flag for frontend routing
    clinical_summary: str
    assessment_level: str
    pathologies: Optional[List[PathologyResult]] = None  # MODIFIED: Optional for normal cases
    heatmap_b64: Optional[str] = None
    ai_explanation: Optional[str] = None
    api_used: Optional[str] = None
    processing_time_ms: float

# NEW: Pydantic models for doctor finder
class FindDoctorsRequest(BaseModel):
    latitude: float
    longitude: float
    pathologies: List[Dict]

class DoctorInfo(BaseModel):
    name: str
    hospital: str      # Same as name — clinic/hospital label for card
    address: str
    distance: str      # Formatted string e.g. "1.2 km"
    phone: str
    website: str
    rating: Optional[float] = None

class FindDoctorsResponse(BaseModel):
    success: bool
    specialist_type: Optional[str] = None
    primary_pathology: Optional[str] = None
    specialists: Optional[List[DoctorInfo]] = None
    general_practitioners: Optional[List[DoctorInfo]] = None
    fallback_message: Optional[str] = None
    generic_advice: Optional[List[str]] = None

# Initialize FastAPI
app = FastAPI(title="Chest X-ray Analysis API", version="2.0.0")  # Version bumped

# Add CORS middleware for React frontend
# ALLOWED_ORIGINS env var: comma-separated list of allowed origins.
# Defaults to localhost for local dev. On HF Spaces, set this to your Vercel URL.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins: list[str] = [
    o.strip() for o in _raw_origins.split(",") if o.strip()
] if _raw_origins else [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model: Any = None
device: Any = None
binary_pipeline: Any = None
heatmap_available: bool = False

def load_models():
    """Load all models on startup"""
    global model, device, binary_pipeline, heatmap_available

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CheXNet model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = collections.OrderedDict()
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    model = DenseNet121(14, True)
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    print("CheXNet model loaded")

    # Load binary pipeline
    models_dir = os.path.join(BASE_DIR, 'models', 'binary')
    binary_pipeline = BinaryClassifierPipeline(
        model1_path=os.path.join(models_dir, 'garbage_vs_xray.pth'),
        model2_path=os.path.join(models_dir, 'chest_vs_other.pth'),
        model3_path=os.path.join(models_dir, 'normal_vs_abnormal.pth')
    )
    print("Binary pipeline loaded")

    heatmap_available = True
    print("Heatmap generation enabled")

# Load models on startup
load_models()

# Preprocessing transforms
preprocess = T.Compose([
    T.Resize(256),
    T.TenCrop(224),
    T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
    T.Lambda(lambda crops: torch.stack([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
        for crop in crops
    ]))
])

def classify_confidence(probability: float) -> str:
    """Classify prediction confidence level"""
    if probability >= 0.65:
        return 'HIGH'
    elif probability >= 0.40:
        return 'MEDIUM'
    elif probability >= 0.20:
        return 'LOW'
    else:
        return 'MINIMAL'

def get_clinical_summary(results: List[Tuple[str, float]], model3_normal: bool) -> Tuple[str, str]:
    """Generate clinical interpretation summary with clear normal/abnormal decisions"""
    high_findings = [(name, prob) for name, prob in results if prob >= 0.65]
    medium_findings = [(name, prob) for name, prob in results if 0.40 <= prob < 0.65]

    if model3_normal and len(high_findings) == 0 and len(medium_findings) == 0:
        return "✅ NORMAL - No significant abnormalities detected", "NORMAL"

    if model3_normal and len(medium_findings) > 0:
        diseases = ", ".join([name for name, _ in medium_findings])
        return f"⚠️ LOW CONFIDENCE - Minor findings: {diseases}. Clinical correlation recommended.", "BORDERLINE"

    if not model3_normal or len(high_findings) > 0:
        diseases = ", ".join([name for name, _ in high_findings or medium_findings])
        return f"⚠️ ABNORMAL - Significant findings: {diseases}. Further evaluation required.", "ABNORMAL"

    # Default case
    return "Analysis completed - Review detailed results below", "UNKNOWN"

def predict_chexnet(image: Image.Image) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
    """Run CheXNet prediction"""
    arr = np.stack([np.array(image)]*3, axis=-1).astype(np.uint8)
    inp = preprocess(Image.fromarray(arr))
    inp = inp.to(device) # type: ignore

    with torch.no_grad():
        out = model(inp) # type: ignore
        probs = torch.sigmoid(out)
        probs_mean = probs.mean(dim=0)

    pairs = sorted(
        zip(CLASS_NAMES, probs_mean.cpu().numpy().tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return pairs, inp[0].unsqueeze(0)

def generate_heatmap_b64(image_rgb: Image.Image, inp: torch.Tensor) -> Optional[str]:
    """Generate Grad-CAM heatmap and return as base64"""
    if not heatmap_available:
        return None

    try:
        heatmap_img_np = generate_gradcam(model, inp, device, image_rgb, 0.5) # type: ignore
        if heatmap_img_np is None:
            return None
        
        # Convert numpy array to PIL Image
        heatmap_pil = Image.fromarray(heatmap_img_np)
        
        # Convert to base64
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Heatmap generation failed: {e}")
        return None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)):
    """
    Analyze uploaded X-ray image with modified decision flow:
    - Normal → Return health tips (skip CheXNet)
    - Abnormal → Run full CheXNet analysis
    """
    import time
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload PNG, JPG, or TIFF images.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image_rgb = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run binary pipeline with cleaner messages
        binary_results = []
        validation_result = binary_pipeline.validate(image_rgb) # type: ignore

        # Model 1: Garbage vs X-ray (keep brief)
        model1_result = validation_result['results']['model1']
        binary_results.append(BinaryResult(
            model="Image Validation",
            is_valid=model1_result.get('is_xray', False) if model1_result else False,
            confidence=model1_result.get('confidence', 0.0) if model1_result else 0.0,
            message="Valid X-ray image" if (model1_result and model1_result.get('is_xray')) else "Invalid image type"
        ))

        # Model 2: Chest vs Other (keep brief)
        model2_result = validation_result['results']['model2']
        binary_results.append(BinaryResult(
            model="Anatomy Check",
            is_valid=model2_result.get('is_chest', False) if model2_result else False,
            confidence=model2_result.get('confidence', 0.0) if model2_result else 0.0,
            message="Chest X-ray confirmed" if (model2_result and model2_result.get('is_chest')) else "Not a chest X-ray"
        ))

        # Model 3: Normal vs Abnormal (key for decision making)
        model3_result = validation_result['results']['model3']
        is_normal = model3_result.get('prediction', '').lower() == 'normal' if model3_result else False
        binary_results.append(BinaryResult(
            model="Initial Assessment",
            is_valid=True,  # Always proceed for analysis
            confidence=model3_result.get('confidence', 0.0) if model3_result else 0.0,
            message="Normal findings" if is_normal else "Abnormal findings detected"
        ))

        valid_for_analysis = validation_result['valid']

        if not valid_for_analysis:
            return AnalysisResponse(
                binary_pipeline=binary_results,
                valid_for_analysis=False,
                is_normal=False,  # NEW FIELD
                clinical_summary=validation_result['message'],
                assessment_level="INVALID",
                pathologies=None,  # Changed from [] to None
                heatmap_b64=None,
                ai_explanation="Image validation failed. Please upload a valid chest X-ray image.",
                api_used="System Message",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # ====================================================================
        # MODIFIED DECISION FLOW - CRITICAL CHANGE
        # ====================================================================
        
        # CASE A: NORMAL — Skip CheXNet, return health tips
        if validation_result.get('skip_main_model', False) or is_normal:
            print("📋 NORMAL case detected — Generating health tips (skipping CheXNet)")
            
            # Generate AI explanation for normal case
            ai_result = explainability_ai.generate_normal_explanation(model3_result or {})
            
            processing_time = (time.time() - start_time) * 1000
            
            return AnalysisResponse(
                binary_pipeline=binary_results,
                valid_for_analysis=True,
                is_normal=True,  # NEW: Critical flag
                clinical_summary="✅ NORMAL - No significant abnormalities detected",
                assessment_level="NORMAL",
                pathologies=None,  # No pathology analysis for normal cases
                heatmap_b64=None,  # No heatmap for normal cases
                ai_explanation=ai_result['explanation'],
                api_used=ai_result['api_used'],
                processing_time_ms=round(processing_time, 1)
            )
        
        # CASE B: ABNORMAL — Proceed to full CheXNet analysis
        print("🔬 ABNORMAL case detected — Running full CheXNet analysis")
        
        # Run CheXNet analysis
        results, inp = predict_chexnet(image)

        # Get model3 normal status for summary
        model3_normal = False  # Already determined to be abnormal

        summary_text, assessment_level = get_clinical_summary(results, model3_normal)

        # Prepare pathology results
        pathologies = []
        for disease, prob in results:
            confidence = classify_confidence(prob)
            is_positive = prob >= CLINICAL_THRESHOLDS.get(disease, 0.5)
            decision = "POSITIVE" if is_positive else "NEGATIVE"
            pathologies.append(PathologyResult(
                pathology=disease,
                probability=round(prob, 4),
                confidence_level=confidence,
                clinical_decision=decision
            ))

        # Generate heatmap
        heatmap_b64 = generate_heatmap_b64(image_rgb, inp)

        # Generate AI explanation for abnormal case
        main_model_result = {
            'assessment_level': assessment_level,
            'pathologies': [{'name': p.pathology, 'probability': p.probability} for p in pathologies]
        }

        ai_result = explainability_ai.generate_abnormal_explanation(
            model3_result or {}, 
            main_model_result
        )

        processing_time = (time.time() - start_time) * 1000

        return AnalysisResponse(
            binary_pipeline=binary_results,
            valid_for_analysis=True,
            is_normal=False,  # NEW: Critical flag
            clinical_summary=summary_text,
            assessment_level=assessment_level,
            pathologies=pathologies,
            heatmap_b64=heatmap_b64,
            ai_explanation=ai_result['explanation'],
            api_used=ai_result['api_used'],
            processing_time_ms=round(processing_time, 1)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# NEW ENDPOINT
@app.post("/find-doctors", response_model=FindDoctorsResponse)
async def find_doctors(request: FindDoctorsRequest = Body(...)):
    """
    Find specialized doctors near user location based on detected pathologies.
    
    Request body:
    {
        "latitude": 16.5062,
        "longitude": 80.6480,
        "pathologies": [
            {"name": "Pneumonia", "probability": 0.85},
            {"name": "Infiltration", "probability": 0.62}
        ]
    }
    """
    try:
        # Use Geoapify service to find doctors
        result = geoapify_finder.find_doctors_for_pathology(
            latitude=request.latitude,
            longitude=request.longitude,
            pathologies=request.pathologies
        )
        
        # Check if we got results or need to use fallback
        if not result.get('specialists') and not result.get('general_practitioners'):
            fallback = geoapify_finder.get_fallback_recommendations()
            return FindDoctorsResponse(
                success=False,
                fallback_message=fallback.get('fallback_message'),
                generic_advice=fallback.get('generic_advice')
            )
        
        # Format successful response — map geoapify fields to frontend-compatible DoctorInfo
        def _to_doctor_info(doc: dict) -> DoctorInfo:
            return DoctorInfo(
                name=doc.get('name', 'Medical Facility'),
                hospital=doc.get('name', 'Medical Facility'),  # use name as hospital label
                address=doc.get('address', 'Address not available'),
                distance=f"{doc.get('distance_km', 0.0)} km",
                phone=doc.get('phone', 'Not available'),
                website=doc.get('website', 'Not available'),
                rating=doc.get('rating'),
            )

        specialists = [_to_doctor_info(doc) for doc in result.get('specialists', [])]
        gps = [_to_doctor_info(doc) for doc in result.get('general_practitioners', [])]
        
        return FindDoctorsResponse(
            success=True,
            specialist_type=result.get('specialist_type'),
            primary_pathology=result.get('primary_pathology'),
            specialists=specialists,
            general_practitioners=gps,
            fallback_message=None,
            generic_advice=None
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in find_doctors endpoint: {e}")
        
        # Return fallback on error
        fallback = geoapify_finder.get_fallback_recommendations()
        return FindDoctorsResponse(
            success=False,
            fallback_message=fallback.get('fallback_message'),
            generic_advice=fallback.get('generic_advice')
        )


@app.get("/health")
async def health_check():
    """Health check endpoint - ENHANCED"""
    return {
        "status": "healthy",
        "models_loaded": model is not None and binary_pipeline is not None,
        "gemini_available": explainability_ai.client is not None,  # NEW
        "geoapify_available": geoapify_finder.api_key is not None,  # NEW
    }

if __name__ == "__main__":
    import uvicorn
    # PORT env var defaults to 8000 locally; HF Spaces sets it to 7860 via Dockerfile
    _port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=_port)
