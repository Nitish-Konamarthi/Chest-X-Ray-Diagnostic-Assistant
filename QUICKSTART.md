"""
QUICK START GUIDE
================

This document provides step-by-step instructions to run the CheXNet Medical AI System.

## Prerequisites

✅ Verify all prerequisites are installed:
- Python 3.8+
- Node.js 16+
- CUDA 11.8+ (optional, for GPU acceleration)

## 1. Backend Setup (Python/FastAPI)

### Step 1: Install Python Dependencies
cd backend
pip install -r requirements.txt

### Step 2: Configure API Keys (Optional)
Create or edit .env file in project root:
```
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Step 3: Start Backend Server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

Expected output:
✅ Using device: cuda (or cpu)
✅ CheXNet model loaded
✅ Binary pipeline loaded
✅ Heatmap generation enabled
✅ Uvicorn running on http://0.0.0.0:8000

Backend is ready when you see: "Application startup complete"

## 2. Frontend Setup (React)

### Step 1: Install Node Dependencies
cd frontend
npm install

### Step 2: Start Frontend Development Server
npm start

Expected output:
✅ Compiled successfully!
✅ Local: http://localhost:3000

Open http://localhost:3000 in your browser

## 3. Streamlit Setup (Alternative Backup)

### Step 1: Install Streamlit (in backend venv)
pip install streamlit

### Step 2: Start Streamlit App
streamlit run chexnet/streamlit_app.py

Expected output:
✅ You're running the latest version of Streamlit
✅ Open http://localhost:8501

## 4. System Testing

### Quick Sanity Check
python test_integration.py

Expected: All ✅ tests pass

### Test API Endpoint
curl -X GET http://localhost:8000/health

Expected response:
{
  "status": "healthy",
  "models_loaded": true
}

## Usage Workflow

### Via React UI (Recommended)
1. Open http://localhost:3000
2. Click "Choose X-ray Image"
3. Select a chest X-ray image
4. Click "Analyze Image"
5. View results including:
   - Image validation results
   - AI-powered clinical assessment
   - Grad-CAM++ heatmap
   - Explainable AI insights
   - Disease predictions

### Via Streamlit (Backup)
1. Open http://localhost:8501
2. Upload chest X-ray image
3. Adjust settings in sidebar (optional)
4. Click "Run CheXNet Pathology Analysis"
5. View results with heatmap and diagnosis

### Via API (Direct)
POST http://localhost:8000/analyze
Content-Type: multipart/form-data

Response includes:
```json
{
  "binary_pipeline": [...],
  "valid_for_analysis": true,
  "clinical_summary": "...",
  "assessment_level": "NORMAL|ABNORMAL|BORDERLINE",
  "pathologies": [...],
  "heatmap_b64": "data:image/png;base64,...",
  "ai_explanation": "Clinical explanation from LLM",
  "processing_time_ms": 245.3
}
```

## Troubleshooting

### Problem: CUDA Out of Memory
Solution:
- Use CPU instead: Set CUDA_VISIBLE_DEVICES=-1
- Reduce batch size in backend

### Problem: Models Not Found
Solution:
- Verify paths: python test_integration.py
- Check backend/models/ directory structure

### Problem: Heatmap Not Generating
Solution:
- Expected behavior: heatmap_b64 will be null if generation fails
- Frontend will display image without heatmap
- Check backend logs for errors

### Problem: API Connection Failed (Frontend)
Solution:
- Verify backend is running on port 8000
- Check CORS settings in backend/app.py
- Browser console will show specific error

### Problem: Binary Pipeline Validation Failed
Solution:
- Ensure image is a valid chest X-ray
- Try different image format (PNG, JPG)
- Check backend logs for validation details

### Problem: No AI Explanations
Solution:
- Set API keys in .env file
- Both APIs optional: system works without them
- Check .env file format: KEY=value (no quotes)

## Model Information

### Main Model: CheXNet (DenseNet121)
- Architecture: 121-layer DenseNet
- Classes: 14 chest pathologies
- Model file: backend/models/chexnet/m-30012020-104001.pth.tar
- Size: ~30MB
- Speed: ~200-300ms per image (GPU)

### Binary Pipeline Models:
1. **Model 1**: Garbage vs X-ray (MobileNetV2) - ~14MB
2. **Model 2**: Chest vs Other X-rays (ResNet18) - ~45MB
3. **Model 3**: Normal vs Abnormal (ResNet50) - ~98MB

### Detected Pathologies:
1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

## Advanced Configuration

### Backend Settings (backend/app.py)
```python
MODEL_PATH = "models/chexnet/m-30012020-104001.pth.tar"
CLINICAL_THRESHOLDS = {disease: threshold}  # Per-disease thresholds
CORS allow_origins = ["http://localhost:3000"]  # Frontend URL
```

### Streamlit Settings (chexnet/streamlit_app.py)
```python
high_conf_threshold = 0.65  # High confidence
medium_conf_threshold = 0.40  # Medium confidence
low_conf_threshold = 0.20  # Low confidence
heatmap_blend_alpha = 0.5  # Heatmap intensity
```

## Performance Metrics

### Inference Speed
- CPU: 1000-1500ms per image
- GPU (NVIDIA): 200-300ms per image
- Binary pipeline: 50-100ms

### Model Accuracy (from research)
- Overall AUC: 0.84
- Sensitivity: 77.3%
- Specificity: 85.1%

### Memory Requirements
- GPU: 4GB+ recommended
- CPU: 8GB+ minimum
- Disk: 200MB+ for models

## Support & Documentation

For detailed information:
- See README.md for full documentation
- Check backend/app.py for API details
- Review frontend/src/App.js for UI logic
- Check chexnet/streamlit_app.py for Streamlit setup

## Security Notes

⚠️ Medical Disclaimer:
This system is for educational and research purposes only.
Do not use for actual medical diagnosis without professional review.

✅ Privacy:
- Images are not stored or transmitted externally
- All processing is local
- API keys stored securely in environment variables

## Next Steps

1. ✅ System is set up and running
2. ✅ All models are loaded and working
3. ✅ Both UI backends (React + Streamlit) are functional
4. ✅ Heatmaps are generating correctly
5. ✅ Binary pipeline is validating images
6. ✅ AI explanations are available (if APIs configured)

Happy diagnosing! 🩺🤖
"""