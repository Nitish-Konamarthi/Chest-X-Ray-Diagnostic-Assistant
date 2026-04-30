# 🫁 Chest X-Ray Diagnostic Assistant

An AI-powered medical diagnostic system for chest X-ray analysis, designed to be easy for developers to integrate. This package provides an end-to-end pipeline that validates X-rays, detects 14 different pathologies, generates visual heatmaps, provides clinical AI explanations, and finds nearby doctors.

**Available on both PyPI and NPM:**
- Python (PyPI): `pip install chest-xray-diagnosis`
- Node.js (NPM): `npx chest-xray-diagnosis serve`

---

## 🚀 Features & What Happens Under the Hood

When you feed an image into this package, it automatically processes through 4 stages:

1. **3-Stage Binary Pipeline (Validation)**
   - **Model 1 (MobileNetV2):** Garbage vs. X-ray. Rejects non-medical images instantly.
   - **Model 2 (ResNet18):** Chest vs. Other. Confirms the X-ray is actually of a chest.
   - **Model 3 (ResNet18):** Normal vs. Abnormal. Performs an initial check. If Normal, it skips the heavy processing and generates health tips!

2. **CheXNet Analysis (DenseNet-121)**
   - If the image is Abnormal, it runs through CheXNet.
   - Detects 14 pathologies (Pneumonia, Cardiomegaly, Effusion, Mass, etc.) with probabilities and clinical thresholds.
   - **Grad-CAM++ Heatmaps:** Automatically generates a visual heatmap (returned as a base64 PNG) showing exactly which region of the lungs the AI focused on.

3. **Gemini AI Explainability**
   - Passes the clinical findings to **Google Gemini 3.1 Flash Lite**.
   - Generates a warm, patient-friendly clinical explanation detailing symptoms to watch for and recommended precautions.
   - *(Works without an API key using a comprehensive rule-based fallback system).*

4. **Nearby Doctor Recommendations (Geoapify)**
   - Identifies the dominant pathology (e.g., Pneumonia).
   - Maps the pathology to a specialist (e.g., Pulmonologist).
   - Searches the Geoapify Places API to return real doctors and clinics near the user's provided GPS coordinates.

---

## 🧰 Tech Stack
- **Core AI:** PyTorch, TorchVision
- **Models:** DenseNet-121, ResNet-18, MobileNet-V2
- **Image Processing:** OpenCV, Pillow (PIL)
- **API Server:** FastAPI, Uvicorn
- **External Integrations:** Google GenAI (Gemini), Geoapify Places API

---

## 🔑 API Keys

**The package is fully functional WITHOUT any API keys!** The ML models (binary pipeline + CheXNet + heatmaps) are 100% local. 
However, for the best experience (AI explanations and real doctor search), you should provide keys.

You can configure keys in 4 ways:
1. Interactive CLI Wizard: `chest-xray-diagnosis setup`
2. Environment Variables: `GEMINI_API_KEY` and `GEOAPIFY_API_KEY`
3. `.env` file in your current working directory
4. Passed explicitly to the `ChestXRayAnalyzer` constructor in Python

---

## 💻 CLI Usage

Once installed (`pip install chest-xray-diagnosis`), you have access to a powerful CLI.

### 1. Configure your keys
```bash
chest-xray-diagnosis setup
```
*Interactive prompt that securely saves your Gemini and Geoapify keys to `~/.chexray-diagnosis/.env`.*

### 2. Pre-download models
```bash
chest-xray-diagnosis download-models
```
*Downloads the ~230MB of model weights from HuggingFace Hub to your local cache.*

### 3. Analyze a single image directly in your terminal
```bash
chest-xray-diagnosis analyze path/to/chest_xray.png
```
**Output Example:**
```text
============================================================
  Chest X-Ray Analysis Result
============================================================
  Status    : ⚠️ ABNORMAL — Significant findings: Infiltration, Effusion
  Level     : ABNORMAL
  Normal    : False
  Time      : 343 ms

  Pathology              Prob   Confidence Decision
  ---------------------- ------ ---------- ----------
  Infiltration           78.2%  HIGH       POSITIVE
  Effusion               65.4%  HIGH       POSITIVE
  Atelectasis            41.0%  MEDIUM     NEGATIVE
  ...

  AI Explanation (Gemini 3.1 Flash Lite):
  --------------------------------------------------
  ## 🩺 Important Findings in Your Chest X-ray
  Your chest X-ray shows signs of **Infiltration**...
```
*(You can also use `--json` to output the raw JSON data).*

### 4. Start the FastAPI Server
```bash
chest-xray-diagnosis serve --port 8000
```
*Starts a production-ready API server with CORS enabled. View the interactive Swagger docs at `http://localhost:8000/docs`.*

---

## 🐍 Programmatic Usage (Python SDK)

If you are building your own Python backend, you can import the pipeline directly:

```python
from chexray_diagnosis import ChestXRayAnalyzer

# Initializes the models (auto-downloads weights on first run)
analyzer = ChestXRayAnalyzer(
    gemini_api_key="your-key-here",    # Optional: reads from env if not provided
    geoapify_api_key="your-key-here"   # Optional: reads from env if not provided
)

# Run end-to-end analysis
result = analyzer.analyze("patient_xray.png")

print(result["is_normal"])           # True/False
print(result["clinical_summary"])    # "⚠️ ABNORMAL — Significant findings..."
print(result["pathologies"])         # List of 14 pathologies + probabilities
print(result["ai_explanation"])      # Markdown text from Gemini
print(result["heatmap_b64"])         # Base64 encoded PNG heatmap string
```

---

## ⚠️ Medical Disclaimer

This system is designed for **research and educational purposes only**. It must not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for clinical decisions.

## 📄 License
MIT License.