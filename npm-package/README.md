# chest-xray-diagnosis (NPM)

An AI-powered medical diagnostic system for chest X-ray analysis, designed to be launched instantly via NPM.

This is a **Node.js wrapper** around the Python `chest-xray-diagnosis` package. It gives front-end and Node developers zero-config access to a powerful PyTorch X-ray diagnostic backend.

---

## 🚀 Quick Start (Zero Config)

Run the API server instantly:

```bash
npx chest-xray-diagnosis serve --port 8000
```

### What happens under the hood?
1. Detects if you have **Python 3.9+** installed.
2. Automatically creates an isolated virtual environment at `~/.chexray-diagnosis/venv/`.
3. Installs the core Python machine learning libraries (PyTorch, FastAPI, etc.).
4. Downloads the pre-trained X-ray model weights (~230 MB) from HuggingFace Hub.
5. Launches the REST API server!

---

## 📋 Prerequisites

- **Node.js 16+**
- **Python 3.9+** installed on your system
- (Optional) API Keys:
  - [Gemini API Key](https://makersuite.google.com/app/apikey) (for AI clinical explanations)
  - [Geoapify API Key](https://www.geoapify.com/) (for nearby doctor search)

*Note: The core ML analysis (disease detection & heatmaps) works 100% locally without any API keys!*

---

## 💻 CLI Commands

You can use `npx` to run any of the underlying CLI commands:

### Start the API Server
```bash
npx chest-xray-diagnosis serve --port 8000
```

### Configure API Keys (Interactive Wizard)
```bash
npx chest-xray-diagnosis setup
```
*Securely saves your API keys so the local server can use Gemini and Geoapify.*

### Pre-Download Models
```bash
npx chest-xray-diagnosis download-models
```
*Manually downloads the PyTorch model weights to your local cache.*

### Analyze a Single Image from Terminal
```bash
npx chest-xray-diagnosis analyze path/to/xray.png
```

---

## 🧠 What it does

When an image is sent to the server, it runs through a powerful 4-stage pipeline:
1. **Binary Pipeline:** Validates the image (is it garbage? is it a chest?).
2. **CheXNet (DenseNet-121):** Detects 14 different lung pathologies (Pneumonia, Mass, Effusion, etc.).
3. **Grad-CAM++:** Generates a visual heatmap (returned as base64) highlighting the diseased areas of the lung.
4. **Gemini AI:** Generates a patient-friendly clinical explanation of the findings.

---

## License & Disclaimer
**MIT License.**

**Medical Disclaimer:** For research and educational use only. Not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
