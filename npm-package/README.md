# chest-xray-diagnosis (npm)

AI-powered chest X-ray diagnostic assistant — launch with a single command.

This is a **thin npm wrapper** around the Python `chest-xray-diagnosis` package. It auto-installs Python dependencies in an isolated virtualenv and starts the API server.

## Quick Start

```bash
npx chest-xray-diagnosis serve
```

## Prerequisites

- **Node.js 16+**
- **Python 3.9+** (auto-detected)
- API Keys (configured via `setup` command):
  - [Gemini API Key](https://makersuite.google.com/app/apikey)
  - [Geoapify API Key](https://www.geoapify.com/)

## Commands

```bash
# Start the API server
npx chest-xray-diagnosis serve --port 8000

# Analyze a single image
npx chest-xray-diagnosis analyze path/to/xray.png

# Configure API keys
npx chest-xray-diagnosis setup

# Pre-download models (~230 MB)
npx chest-xray-diagnosis download-models
```

## What happens under the hood

1. Checks for Python 3.9+
2. Creates virtualenv at `~/.chexray-diagnosis/venv/`
3. `pip install chest-xray-diagnosis` inside that venv
4. Downloads model weights from HuggingFace Hub (first run only)
5. Starts the FastAPI server

## License

MIT — See [LICENSE](../LICENSE) for details.

**Medical Disclaimer:** For research and educational use only. Not a substitute for professional medical diagnosis.
