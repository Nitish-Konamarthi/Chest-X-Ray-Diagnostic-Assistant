# 🫁 Chest X-Ray Diagnostic Assistant

An AI-powered medical diagnostic system for chest X-ray analysis with explainable AI, Grad-CAM++ heatmaps, and nearby doctor recommendations.

## 🚀 Features

- **3-Stage Binary Pipeline**: Validates images through garbage detection → chest confirmation → normal/abnormal assessment
- **14-Pathology Detection**:  (DenseNet-121) detects conditions like Pneumonia, Cardiomegaly, Effusion, Mass, Nodule, and more
- **Grad-CAM++ Heatmaps**: Visual overlays showing which regions the AI focused on
- **Gemini AI Explainability**: `gemini-3.1-flash-lite` generates professional clinical explanations in natural language
- **Nearby Doctor Recommendations**: Geoapify-powered specialist finder based on detected pathologies and the user's real location
- **Modular React Frontend**: Clean, component-based UI with a premium medical design

## 🏗️ Architecture

```
Frontend (React)  →  Backend (FastAPI)  →  AI Models         →  External APIs
     ↓                     ↓                    ↓                      ↓
  7 Components        REST API :8000       Binary Pipeline       Gemini 3.1 Flash Lite
  Medical UI          Image Processing      (DenseNet)           Geoapify Places
  Responsive UX       Heatmap Generation   Grad-CAM++ Maps
```

## 📁 Project Structure

```
chexnet-proto/
├── backend/
│   ├── app.py                    # FastAPI application (main entry point)
│   ├── requirements.txt          # Python dependencies
│   └── src/
│       ├── binary_pipeline.py    # 3-stage X-ray validation pipeline
│       ├── HeatmapGenerator.py   # Grad-CAM++ heatmap generation
│       ├── DensenetModels.py     # CheXNet model definition
│       ├── explainability_ai.py  # Gemini AI clinical explanation
│       └── geoapify_service.py   # Nearby doctor search (Geoapify)
├── frontend/
│   └── src/
│       ├── App.js                # Root component & analysis orchestrator
│       ├── App.css               # Global styles
│       └── components/
│           ├── Header.js             # App header
│           ├── UploadSection.js      # X-ray upload & drag-drop
│           ├── BinaryPipelineStatus.js  # Pipeline validation display
│           ├── ImagePanel.js         # Original + heatmap side-by-side
│           ├── PathologyResults.js   # 14-pathology confidence grid
│           ├── AIExplanationCard.js  # Gemini AI explanation renderer
│           └── NearbyDoctors.js      # Geoapify-powered doctor finder
├── chexnet/                      # Original reference repo
├── data/                         # Dataset directory
├── scripts/                      # Utility & training scripts
└── .env                          # API keys (not committed)
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU *(recommended for faster inference)*
- Gemini API Key — [Get one here](https://makersuite.google.com/app/apikey)
- Geoapify API Key — [Get one here](https://www.geoapify.com/)

## 🔧 Setup

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Environment Variables

Create a `.env` file in the **root directory**:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEOAPIFY_API_KEY=your_geoapify_api_key_here
```

> **Note:** OpenAI/ChatGPT is no longer used. The project exclusively uses `gemini-2.0-flash` for AI explanations.

### 4. Run the Application

```bash
# Terminal 1 — Backend
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm start
```

Visit: **http://localhost:3000**

## 🔍 How It Works

### Stage 1 — Binary Pipeline (Validation)
| Model | Task | Result |
|-------|------|--------|
| Model 1 | Garbage vs. X-ray | Rejects non-medical images |
| Model 2 | Chest vs. Other X-ray | Confirms it's a chest X-ray |
| Model 3 | Normal vs. Abnormal | Initial pathology flag |

### Stage 2 — CheXNet Analysis
- Runs DenseNet-121 trained on ChestX-ray14 dataset
- Detects 14 pathologies with confidence scores and clinical thresholds
- Generates **Grad-CAM++** attention heatmaps

### Stage 3 — AI Explainability (Gemini)
- Combines Model 3 + CheXNet results into a structured prompt
- Sends to **Gemini 3.1 Flash Lite** for clinical explanation
- Returns formatted explanation with sections: Summary, Findings, Recommendations

### Stage 4 — Nearby Doctor Finder (Geoapify)
- Detects dominant pathology from results
- Maps pathology → specialist type (e.g., Pneumonia → Pulmonologist)
- Uses browser geolocation to search for real nearby clinics
- Falls back to General Physician if no specialist found

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Upload X-ray image for full analysis |

### Sample Response (`/analyze`)

```json
{
  "binary_pipeline": [
    { "model": "Image Validation", "is_valid": true, "confidence": 0.98, "message": "Valid chest X-ray" }
  ],
  "valid_for_analysis": true,
  "clinical_summary": "✅ NORMAL - No significant abnormalities detected",
  "assessment_level": "NORMAL",
  "pathologies": [
    { "name": "Pneumonia", "confidence": 0.12, "is_detected": false }
  ],
  "heatmap_b64": "data:image/png;base64,...",
  "ai_explanation": "## Summary\n...",
  "doctor_recommendations": [...],
  "processing_time_ms": 245.3
}
```

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vanilla CSS |
| Backend | FastAPI, Uvicorn |
| ML Models | PyTorch, DenseNet-121 (CheXNet) |
| AI Explainability | Google Gemini 3.1 Flash Lite (`google-genai`) |
| Doctor Finder | Geoapify Places API |
| Heatmaps | Grad-CAM++, OpenCV |

## ⚡ Performance

- Binary pipeline: ~50ms
- CheXNet + heatmap: ~150–250ms
- Gemini AI explanation: ~1–3s (network dependent)
- GPU acceleration supported (CUDA)

## 🔒 Privacy & Security

- All image processing is local — no images sent to external servers
- Only text summaries are sent to Gemini for explanation
- API keys stored in `.env` (excluded from version control via `.gitignore`)
- No patient data is logged or persisted

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| Model files not found | Ensure `.pth` files are in `backend/models/` |
| CUDA errors | Set `device = "cpu"` in pipeline files |
| Gemini API errors | Verify `GEMINI_API_KEY` in `.env` |
| Geoapify returns no results | Check `GEOAPIFY_API_KEY` and confirm valid category names |
| Frontend can't reach backend | Confirm backend is running on port `8000` |
| CORS errors | Check `allow_origins` in `app.py` FastAPI config |

## ⚠️ Medical Disclaimer

This system is designed for **research and educational purposes only**. It must not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for clinical decisions.

## 📄 License

For educational and research use only. See individual model licenses for CheXNet model weights.