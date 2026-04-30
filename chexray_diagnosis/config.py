"""
Centralized configuration and secure API-key management.

Resolution order (highest priority first):
    1. Explicit function / constructor arguments
    2. Environment variables  (GEMINI_API_KEY, GEOAPIFY_API_KEY)
    3. .env file in the current working directory
    4. ~/.chexray-diagnosis/.env  (user home config)
    5. Interactive prompt  (CLI mode only — never in library mode)

No key is ever hardcoded, logged in full, or bundled in the package.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Where models are cached after download
CACHE_DIR = Path.home() / ".chexray-diagnosis"
MODELS_DIR = CACHE_DIR / "models"

# HuggingFace defaults
DEFAULT_HF_REPO_ID = "NitishKvv/chexnet-models"

# Model file paths (relative inside the HF repo and local cache)
MODEL_FILES = {
    "binary": {
        "garbage_vs_xray":    "binary/garbage_vs_xray.pth",
        "chest_vs_other":     "binary/chest_vs_other.pth",
        "normal_vs_abnormal": "binary/normal_vs_abnormal.pth",
    },
    "chexnet": {
        "main": "chexnet/m-30012020-104001.pth.tar",
    },
}

# 14 CheXNet pathology class names
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Clinical decision thresholds per pathology
CLINICAL_THRESHOLDS = {
    "Atelectasis": 0.45, "Cardiomegaly": 0.55, "Effusion": 0.50,
    "Infiltration": 0.35, "Mass": 0.60, "Nodule": 0.55,
    "Pneumonia": 0.40, "Pneumothorax": 0.50, "Consolidation": 0.45,
    "Edema": 0.55, "Emphysema": 0.60, "Fibrosis": 0.55,
    "Pleural_Thickening": 0.45, "Hernia": 0.70,
}


# ---------------------------------------------------------------------------
# .env loading helpers
# ---------------------------------------------------------------------------

def _load_env_files() -> None:
    """Load .env files in priority order (cwd first, then home config)."""
    # 1. CWD .env
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        load_dotenv(cwd_env, override=False)

    # 2. Home directory config
    home_env = CACHE_DIR / ".env"
    if home_env.is_file():
        load_dotenv(home_env, override=False)


# Run on import so env vars are available immediately
_load_env_files()


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    """Mask an API key for safe logging: show first 4 and last 4 chars."""
    if len(key) <= 10:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def get_gemini_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini API key."""
    key = explicit or os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


def get_geoapify_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve Geoapify API key."""
    key = explicit or os.getenv("GEOAPIFY_API_KEY")
    return key.strip() if key else None


def get_hf_repo_id(explicit: Optional[str] = None) -> str:
    """Resolve HuggingFace repo ID."""
    return (explicit or os.getenv("HF_REPO_ID") or DEFAULT_HF_REPO_ID).strip()


def get_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve HuggingFace token (only needed for private repos)."""
    token = explicit or os.getenv("HF_TOKEN")
    return token.strip() if token else None


# ---------------------------------------------------------------------------
# Interactive setup (CLI only)
# ---------------------------------------------------------------------------

def interactive_setup() -> None:
    """Walk the user through setting up API keys.  Writes to ~/.chexray-diagnosis/.env"""
    print("\n🔧  Chest X-Ray Diagnosis — Setup Wizard\n")
    print("This will save your API keys to:")
    print(f"  {CACHE_DIR / '.env'}\n")

    gemini = input("Gemini API Key (https://makersuite.google.com/app/apikey): ").strip()
    geoapify = input("Geoapify API Key (https://www.geoapify.com/): ").strip()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env_path = CACHE_DIR / ".env"

    lines = []
    if gemini:
        lines.append(f"GEMINI_API_KEY={gemini}")
    if geoapify:
        lines.append(f"GEOAPIFY_API_KEY={geoapify}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Keys saved to {env_path}")
    print("   You can edit this file any time, or set env vars directly.\n")
