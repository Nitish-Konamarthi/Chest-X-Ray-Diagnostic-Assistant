"""
Model weight auto-downloader.

Downloads pre-trained model weights from HuggingFace Hub on first run.
Models are cached in ~/.chexray-diagnosis/models/ so subsequent runs are instant.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from chexray_diagnosis.config import (
    MODELS_DIR,
    MODEL_FILES,
    DEFAULT_HF_REPO_ID,
    get_hf_repo_id,
    get_hf_token,
)


def _ensure_dirs() -> None:
    """Create local model cache directories."""
    (MODELS_DIR / "binary").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "chexnet").mkdir(parents=True, exist_ok=True)


def get_model_path(category: str, name: str) -> Path:
    """Return the local cache path for a model file.

    Args:
        category: 'binary' or 'chexnet'
        name: key inside MODEL_FILES[category]

    Returns:
        Absolute Path to the cached model file.
    """
    relative = MODEL_FILES[category][name]
    return MODELS_DIR / relative


def models_present() -> bool:
    """Return True if all required model files exist locally."""
    for category, files in MODEL_FILES.items():
        for name in files:
            if not get_model_path(category, name).exists():
                return False
    return True


def download_models(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download all model weights from HuggingFace Hub.

    Args:
        repo_id: HF repo (default: NitishKvv/chexnet-models)
        token:   HF access token (only needed for private repos)
        force:   Re-download even if files already exist
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface-hub not installed.  Run:  pip install huggingface-hub")
        sys.exit(1)

    repo_id = get_hf_repo_id(repo_id)
    token = get_hf_token(token)
    _ensure_dirs()

    print(f"📦 Model repo : {repo_id}")
    print(f"📂 Cache dir  : {MODELS_DIR}")
    print(f"🔑 Auth       : {'Token provided' if token else 'Public repo (no token needed)'}")
    print()

    all_files = []
    for category, files in MODEL_FILES.items():
        for name, relative in files.items():
            all_files.append(relative)

    for remote_path in all_files:
        local_path = MODELS_DIR / remote_path

        if local_path.exists() and not force:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Already cached: {remote_path} ({size_mb:.1f} MB)")
            continue

        print(f"  ⬇️  Downloading {remote_path} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=token,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place the file in a cache sub-path; move if needed
            downloaded_path = Path(downloaded)
            if downloaded_path != local_path:
                import shutil
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_path), str(local_path))

            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Downloaded: {remote_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ Failed to download {remote_path}: {e}")
            sys.exit(1)

    print("\n🎉 All models ready!\n")
