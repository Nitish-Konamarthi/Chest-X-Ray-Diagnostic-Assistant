"""
FastAPI application factory for the Chest X-Ray Diagnosis API.

Usage::

    from chexray_diagnosis.server import create_app
    app = create_app()
"""

import io, os, time, base64, collections
from typing import Any, Dict, List, Optional

import torch, numpy as np, torchvision.transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chexray_diagnosis.config import CLASS_NAMES, CLINICAL_THRESHOLDS
from chexray_diagnosis.models.download import download_models, get_model_path, models_present
from chexray_diagnosis.models.densenet import DenseNet121
from chexray_diagnosis.models.binary_pipeline import BinaryClassifierPipeline
from chexray_diagnosis.services.heatmap import generate_gradcam
from chexray_diagnosis.services.explainability import ExplainabilityAI
from chexray_diagnosis.services.geoapify import GeoapifyDoctorFinder
import chexray_diagnosis as pkg


# --- Pydantic schemas ---

class BinaryResult(BaseModel):
    model: str; is_valid: bool; confidence: float; message: str

class PathologyResult(BaseModel):
    pathology: str; probability: float; confidence_level: str; clinical_decision: str

class AnalysisResponse(BaseModel):
    binary_pipeline: List[BinaryResult]; valid_for_analysis: bool; is_normal: bool
    clinical_summary: str; assessment_level: str
    pathologies: Optional[List[PathologyResult]] = None
    heatmap_b64: Optional[str] = None; ai_explanation: Optional[str] = None
    api_used: Optional[str] = None; processing_time_ms: float

class FindDoctorsRequest(BaseModel):
    latitude: float; longitude: float; pathologies: List[Dict]

class DoctorInfo(BaseModel):
    name: str; hospital: str; address: str; distance: str
    phone: str; website: str; rating: Optional[float] = None

class FindDoctorsResponse(BaseModel):
    success: bool; specialist_type: Optional[str] = None
    primary_pathology: Optional[str] = None
    specialists: Optional[List[DoctorInfo]] = None
    general_practitioners: Optional[List[DoctorInfo]] = None
    fallback_message: Optional[str] = None; generic_advice: Optional[List[str]] = None


# --- Helpers ---

def _classify(p):
    if p >= 0.65: return "HIGH"
    if p >= 0.40: return "MEDIUM"
    if p >= 0.20: return "LOW"
    return "MINIMAL"

def _summary(results, m3_normal):
    high = [(n, p) for n, p in results if p >= 0.65]
    med = [(n, p) for n, p in results if 0.40 <= p < 0.65]
    if m3_normal and not high and not med:
        return "✅ NORMAL — No significant abnormalities", "NORMAL"
    if m3_normal and med:
        return f"⚠️ LOW CONFIDENCE — {', '.join(n for n,_ in med)}", "BORDERLINE"
    if not m3_normal or high:
        return f"⚠️ ABNORMAL — {', '.join(n for n,_ in (high or med))}", "ABNORMAL"
    return "Analysis completed", "UNKNOWN"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    if not models_present():
        download_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CheXNet
    ckpt = torch.load(str(get_model_path("chexnet", "main")), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = collections.OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model = DenseNet121(14, True); model.load_state_dict(cleaned, strict=False)
    model.to(device); model.eval()

    bp = BinaryClassifierPipeline(
        str(get_model_path("binary", "garbage_vs_xray")),
        str(get_model_path("binary", "chest_vs_other")),
        str(get_model_path("binary", "normal_vs_abnormal")),
    )
    expl = ExplainabilityAI(); doc_finder = GeoapifyDoctorFinder()

    pre = T.Compose([T.Resize(256), T.TenCrop(224),
        T.Lambda(lambda c: torch.stack([T.ToTensor()(x) for x in c])),
        T.Lambda(lambda c: torch.stack([T.Normalize([.485,.456,.406],[.229,.224,.225])(x) for x in c]))])

    def predict(img):
        arr = np.stack([np.array(img)]*3, axis=-1).astype(np.uint8)
        inp = pre(Image.fromarray(arr)).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(inp)).mean(dim=0)
        return sorted(zip(CLASS_NAMES, probs.cpu().numpy().tolist()), key=lambda x:x[1], reverse=True), inp[0].unsqueeze(0)

    def hm_b64(rgb, inp):
        try:
            hm = generate_gradcam(model, inp, device, rgb, 0.5)
            if hm is None: return None
            buf = io.BytesIO(); Image.fromarray(hm).save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except: return None

    # --- FastAPI ---
    app = FastAPI(title="Chest X-Ray Diagnosis API", version=pkg.__version__)

    origins = os.getenv("ALLOWED_ORIGINS", "")
    app.add_middleware(CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()] if origins else ["http://localhost:3000","http://127.0.0.1:3000","http://localhost:5173","http://127.0.0.1:5173"],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    async def root():
        return {"message": "Chest X-Ray Diagnosis API", "version": pkg.__version__}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "models_loaded": True, "gemini_available": expl.client is not None, "geoapify_available": doc_finder.api_key is not None}

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(file: UploadFile = File(...)):
        start = time.time()
        if not file.filename.lower().endswith((".png",".jpg",".jpeg",".tiff")):
            raise HTTPException(400, "Invalid file type")
        try:
            data = await file.read()
            img = Image.open(io.BytesIO(data)).convert("L")
            rgb = Image.open(io.BytesIO(data)).convert("RGB")
            val = bp.validate(rgb); r = val["results"] or {}
            m1,m2,m3 = r.get("model1"),r.get("model2"),r.get("model3")
            br = [
                BinaryResult(model="Image Validation", is_valid=m1.get("is_xray",False) if m1 else False, confidence=m1.get("confidence",0) if m1 else 0, message="Valid X-ray" if (m1 and m1.get("is_xray")) else "Invalid"),
                BinaryResult(model="Anatomy Check", is_valid=m2.get("is_chest",False) if m2 else False, confidence=m2.get("confidence",0) if m2 else 0, message="Chest confirmed" if (m2 and m2.get("is_chest")) else "Not chest"),
            ]
            is_n = m3.get("prediction","").lower()=="normal" if m3 else False
            br.append(BinaryResult(model="Initial Assessment", is_valid=True, confidence=m3.get("confidence",0) if m3 else 0, message="Normal" if is_n else "Abnormal detected"))

            if not val["valid"]:
                return AnalysisResponse(binary_pipeline=br, valid_for_analysis=False, is_normal=False, clinical_summary=val["message"], assessment_level="INVALID", ai_explanation="Validation failed.", api_used="System", processing_time_ms=(time.time()-start)*1000)

            if val.get("skip_main_model") or is_n:
                ai = expl.generate_normal_explanation(m3 or {})
                return AnalysisResponse(binary_pipeline=br, valid_for_analysis=True, is_normal=True, clinical_summary="✅ NORMAL", assessment_level="NORMAL", ai_explanation=ai["explanation"], api_used=ai["api_used"], processing_time_ms=round((time.time()-start)*1000,1))

            res, inp = predict(img); sm, lv = _summary(res, False)
            paths = [PathologyResult(pathology=d, probability=round(p,4), confidence_level=_classify(p), clinical_decision="POSITIVE" if p>=CLINICAL_THRESHOLDS.get(d,0.5) else "NEGATIVE") for d,p in res]
            hm = hm_b64(rgb, inp)
            mr = {"assessment_level": lv, "pathologies": [{"name":p.pathology,"probability":p.probability} for p in paths]}
            ai = expl.generate_abnormal_explanation(m3 or {}, mr)
            return AnalysisResponse(binary_pipeline=br, valid_for_analysis=True, is_normal=False, clinical_summary=sm, assessment_level=lv, pathologies=paths, heatmap_b64=hm, ai_explanation=ai["explanation"], api_used=ai["api_used"], processing_time_ms=round((time.time()-start)*1000,1))
        except Exception as e:
            import traceback; traceback.print_exc()
            raise HTTPException(500, f"Analysis failed: {e}")

    @app.post("/find-doctors", response_model=FindDoctorsResponse)
    async def find_doctors(request: FindDoctorsRequest = Body(...)):
        try:
            res = doc_finder.find_doctors_for_pathology(request.latitude, request.longitude, request.pathologies)
            if not res.get("specialists") and not res.get("general_practitioners"):
                fb = doc_finder.get_fallback_recommendations()
                return FindDoctorsResponse(success=False, fallback_message=fb.get("fallback_message"), generic_advice=fb.get("generic_advice"))
            def ti(d): return DoctorInfo(name=d.get("name","Facility"), hospital=d.get("name","Facility"), address=d.get("address","N/A"), distance=f"{d.get('distance_km',0)} km", phone=d.get("phone","N/A"), website=d.get("website","N/A"), rating=d.get("rating"))
            return FindDoctorsResponse(success=True, specialist_type=res.get("specialist_type"), primary_pathology=res.get("primary_pathology"), specialists=[ti(d) for d in res.get("specialists",[])], general_practitioners=[ti(d) for d in res.get("general_practitioners",[])])
        except:
            fb = doc_finder.get_fallback_recommendations()
            return FindDoctorsResponse(success=False, fallback_message=fb.get("fallback_message"), generic_advice=fb.get("generic_advice"))

    return app
