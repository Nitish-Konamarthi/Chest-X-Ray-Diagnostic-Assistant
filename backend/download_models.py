"""
Model Downloader for Hugging Face Spaces
=========================================
This script downloads all required model files from a HuggingFace Hub repository
at build time (or startup). Models are stored in the local ./models/ directory.

Required HF Secrets:
  - HF_TOKEN  : Your Hugging Face access token (for private repos)
  - HF_REPO_ID: The repo where you uploaded the models (e.g. "your-username/chexnet-models")

Models downloaded:
  Binary pipeline (models/binary/):
    - garbage_vs_xray.pth
    - chest_vs_other.pth
    - normal_vs_abnormal.pth

  CheXNet (models/chexnet/):
    - m-30012020-104001.pth.tar
"""

import os
import sys

def download_models():
    hf_token = os.getenv("HF_TOKEN", "")
    repo_id  = os.getenv("HF_REPO_ID", "")

    if not repo_id:
        print("⚠️  HF_REPO_ID not set — skipping model download.")
        print("   Models must already exist at ./models/ (e.g. mounted volume or manual upload).")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Define model files to download: (remote_filename, local_destination_path)
    model_files = [
        # Binary pipeline models
        ("binary/garbage_vs_xray.pth",          "models/binary/garbage_vs_xray.pth"),
        ("binary/chest_vs_other.pth",            "models/binary/chest_vs_other.pth"),
        ("binary/normal_vs_abnormal.pth",        "models/binary/normal_vs_abnormal.pth"),
        # CheXNet model
        ("chexnet/m-30012020-104001.pth.tar",    "models/chexnet/m-30012020-104001.pth.tar"),
    ]

    os.makedirs("models/binary",  exist_ok=True)
    os.makedirs("models/chexnet", exist_ok=True)

    for remote_path, local_path in model_files:
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✅ Already exists: {local_path} ({size_mb:.1f} MB)")
            continue

        print(f"⬇️  Downloading {remote_path} → {local_path} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",      # Change to "model" if you upload as a Model repo
                token=hf_token or None,
                local_dir=".",            # Download relative to current directory
                local_dir_use_symlinks=False,
            )
            # hf_hub_download caches in a temp dir; move to our expected path
            if downloaded != local_path:
                import shutil
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.move(downloaded, local_path)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✅ Downloaded: {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ Failed to download {remote_path}: {e}")
            print("   The app will fail to start if this model is required.")
            sys.exit(1)

    print("\n🎉 All models downloaded successfully!\n")


if __name__ == "__main__":
    download_models()
