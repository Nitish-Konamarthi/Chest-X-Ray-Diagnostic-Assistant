#!/usr/bin/env python3
"""
Comprehensive verification script for CheXNet app
Tests all components and validates the model works
"""

import os
import sys
import traceback

print("=" * 70)
print("CHEXNET APP VERIFICATION SUITE")
print("=" * 70)

# Test 1: Check Python version
print("\n[1] Python Version Check...")
try:
    version_info = sys.version_info
    print(f"   ✓ Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    assert version_info >= (3, 8), "Python 3.8+ required"
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check all required packages
print("\n[2] Checking Required Packages...")
required_packages = [
    'torch',
    'torchvision',
    'streamlit',
    'fastapi',
    'uvicorn',
    'PIL',
    'numpy',
    'pandas',
    'cv2',
    'sklearn'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'PIL':
            __import__('PIL')
        elif package == 'cv2':
            __import__('cv2')
        elif package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} (MISSING)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ERROR: Missing packages: {', '.join(missing_packages)}")
    print(f"   Run: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 3: Check model file
print("\n[3] Checking Model File...")
model_path = "chexnet/models/m-25012018-123527.pth.tar"
try:
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✓ Model found: {model_path}")
        print(f"   ✓ File size: {size_mb:.1f} MB")
    else:
        print(f"   ✗ Model NOT found at {model_path}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Import core modules
print("\n[4] Testing Module Imports...")
sys.path.insert(0, 'chexnet')

try:
    from DensenetModels import DenseNet121
    print("   ✓ DensenetModels imported")
except Exception as e:
    print(f"   ✗ DensenetModels import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from DatasetGenerator import DatasetGenerator
    print("   ✓ DatasetGenerator imported")
except Exception as e:
    print(f"   ✗ DatasetGenerator import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from ChexnetTrainer import ChexnetTrainer
    print("   ✓ ChexnetTrainer imported")
except Exception as e:
    print(f"   ✗ ChexnetTrainer import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from HeatmapGenerator import HeatmapGenerator
    print("   ✓ HeatmapGenerator imported")
except Exception as e:
    print(f"   ✗ HeatmapGenerator import FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model Loading
print("\n[5] Testing Model Loading...")
try:
    import torch
    import collections
    
    print("   Loading checkpoint...")
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    
    # Verify state dict structure
    print(f"   ✓ Checkpoint loaded")
    print(f"   ✓ State dict keys: {len(state)} entries")
    
    # Create and load model
    print("   Creating model...")
    model = DenseNet121(14, True)
    
    new_state = collections.OrderedDict()
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    
    model.load_state_dict(new_state)
    print("   ✓ Model state loaded successfully")
    
    # Test device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"   ✓ Model on device: {device}")
    
except Exception as e:
    print(f"   ✗ Model loading FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test inference
print("\n[6] Testing Model Inference...")
try:
    import numpy as np
    import torchvision.transforms as T
    from PIL import Image
    
    # Create dummy image
    dummy_img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy_img_array)
    
    # Transform
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229])
    ])
    
    inp = preprocess(dummy_img).unsqueeze(0).to(device)
    
    # Inference
    print("   Running inference...")
    with torch.no_grad():
        output = model(inp)
        probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
    
    print(f"   ✓ Inference successful")
    print(f"   ✓ Output shape: {probs.shape}")
    print(f"   ✓ Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"   ✓ Sample predictions:")
    
    CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
                   'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
                   'Fibrosis','Pleural_Thickening','Hernia']
    
    pairs = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    for i, (label, prob) in enumerate(pairs[:3]):
        print(f"      {i+1}. {label}: {prob*100:.2f}%")
    
except Exception as e:
    print(f"   ✗ Inference FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: App syntax check
print("\n[7] Checking App Syntax...")
import py_compile

app_files = [
    'chexnet/streamlit_app.py',
    'chexnet/DensenetModels.py',
    'chexnet/DatasetGenerator.py',
    'chexnet/ChexnetTrainer.py',
    'chexnet/HeatmapGenerator.py',
    'chexnet/Main.py',
    'app.py'
]

all_valid = True
for app_file in app_files:
    try:
        py_compile.compile(app_file, doraise=True)
        print(f"   ✓ {app_file}")
    except py_compile.PyCompileError as e:
        print(f"   ✗ {app_file} - SYNTAX ERROR")
        print(f"      {e}")
        all_valid = False

if not all_valid:
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✓ ALL VERIFICATION CHECKS PASSED!")
print("=" * 70)
print("\nYour CheXNet app is ready to run:")
print("\n  Option 1 - Web UI (Streamlit):")
print("    streamlit run chexnet/streamlit_app.py")
print("\n  Option 2 - REST API (FastAPI):")
print("    python app.py")
print("\n  Option 3 - Training/Testing (CLI):")
print("    python -m chexnet.Main")
print("\n" + "=" * 70)
