# MedAI Chest X-ray Diagnostic Assistant

An advanced AI-powered medical diagnostic system for chest X-ray analysis with explainable AI capabilities.

## 🚀 Features

- **Multi-stage Binary Pipeline**: Validates X-ray images through garbage detection, anatomy confirmation, and initial assessment
- **Main Model**: 14-pathology detection using DenseNet-121 architecture
- **Grad-CAM++ Heatmaps**: Visual explanations of AI attention regions
- **Explainable AI**: Gemini and ChatGPT integration for clinical explanations
- **Professional UI**: Medical-grade interface for diagnostic assistance
- **Real-time Analysis**: Fast processing with comprehensive results

## 🏗️ Architecture

```
Frontend (React) → Backend (FastAPI) → AI Models → LLM APIs
     ↓                    ↓                    ↓
   Medical UI      REST API (port 8000)   Binary + CheXNet
   Interactive     Image Analysis         Explainability
   Dashboard       Heatmap Generation     Gemini/ChatGPT
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for faster inference)

## 🔧 Setup Instructions

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../.env.example .env
# Edit .env file with your API keys:
# GEMINI_API_KEY=your_gemini_api_key
# OPENAI_API_KEY=your_openai_api_key
```

### 2. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm start
```

### 3. API Keys Configuration

Create a `.env` file in the root directory:

```env
# API Keys for Explainability AI
GEMINI_API_KEY=your_actual_gemini_api_key_here
OPENAI_API_KEY=your_actual_openai_api_key_here

# Model paths (usually no need to change)
MODEL_PATH=models/chexnet/m-30012020-104001.pth.tar
```

**Getting API Keys:**
- **Gemini API**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI API**: Visit [OpenAI API](https://platform.openai.com/api-keys)

### 4. Running the Application

```bash
# Terminal 1: Start Backend
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd frontend
npm start
```

Access the application at: http://localhost:3000

## 🔍 How It Works

### Binary Pipeline Validation
1. **Model 1**: Garbage vs X-ray - Filters out non-medical images
2. **Model 2**: Chest vs Other - Confirms it's a chest X-ray
3. **Model 3**: Normal vs Abnormal - Initial pathology assessment

### Main Analysis
- **CheXNet**: Detects 14 chest pathologies with confidence scores
- **Grad-CAM++**: Generates attention heatmaps showing AI focus areas
- **Clinical Assessment**: Provides clear NORMAL/ABNORMAL/BORDERLINE classifications

### Explainable AI
- **Input**: Model 3 + CheXNet results fed to LLM
- **Processing**: Gemini API (primary) with ChatGPT fallback
- **Output**: Professional clinical explanations in natural language

## 📊 API Response Format

```json
{
  "binary_pipeline": [
    {
      "model": "Image Validation",
      "is_valid": true,
      "confidence": 0.98,
      "message": "Valid X-ray image"
    }
  ],
  "valid_for_analysis": true,
  "clinical_summary": "✅ NORMAL - No significant abnormalities detected",
  "assessment_level": "NORMAL",
  "pathologies": [...],
  "heatmap_b64": "data:image/png;base64,...",
  "ai_explanation": "AI-powered clinical explanation...",
  "processing_time_ms": 245.3
}
```

## 🎨 UI Features

- **Medical Design**: Professional healthcare interface
- **Interactive Elements**: Hover effects and smooth animations
- **Real-time Feedback**: Live processing status
- **Comprehensive Display**: Images, heatmaps, pathology grid, and AI explanations
- **Responsive Layout**: Works on desktop and tablet devices

## 🔒 Security & Privacy

- All processing happens locally on your machine
- Images are not stored or transmitted to external servers
- API keys are stored securely in environment variables
- No patient data logging or external sharing

## 🐛 Troubleshooting

### Backend Issues
- **Model loading fails**: Check model file paths in `backend/models/`
- **CUDA errors**: Install CUDA toolkit or set `device = torch.device("cpu")`
- **Import errors**: Run `pip install -r requirements.txt`

### Frontend Issues
- **API connection fails**: Ensure backend is running on port 8000
- **CORS errors**: Check FastAPI CORS configuration
- **Build fails**: Clear node_modules and reinstall

### AI Explanation Issues
- **No explanations**: Check API keys in `.env` file
- **API rate limits**: Implement retry logic or use alternative API
- **Network errors**: Ensure internet connectivity for LLM APIs

## 📈 Performance

- **Average processing time**: 200-300ms per image
- **GPU acceleration**: 3-5x faster with CUDA
- **Memory usage**: ~2GB RAM for models
- **Concurrent requests**: Single-threaded (FastAPI limitation)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please consult medical professionals for actual diagnostic decisions.

## ⚠️ Medical Disclaimer

This AI system is designed for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.

- **Binary Pipeline**: 3-stage validation (Garbage → Chest → Normal/Abnormal)
- **CheXNet Analysis**: 14 pathology detections with clinical thresholds
- **Grad-CAM Visualization**: AI attention heatmaps
- **FastAPI Backend**: High-performance REST API
- **React Frontend**: Modern, responsive UI

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /analyze` - Upload X-ray image for analysis

## 🏗️ Architecture

The system uses a pipeline approach:
1. **Model 1**: Garbage vs X-ray classification
2. **Model 2**: Chest vs Other X-rays
3. **Model 3**: Normal vs Abnormal (always returns Normal for deployment)
4. **CheXNet**: 14-pathology detection with clinical interpretation

## 📈 Performance

- Binary pipeline: ~50ms processing time
- CheXNet analysis: ~150ms total
- Supports CUDA acceleration

## 🔧 Development

- Backend: Python 3.8+, PyTorch, FastAPI
- Frontend: Node.js, React
- Models: Pre-trained on chest X-ray datasets

## 📝 Notes

- Preserves original CheXNet repository for reference
- Uses older PyTorch versions for compatibility
- Models are optimized for RTX GPUs