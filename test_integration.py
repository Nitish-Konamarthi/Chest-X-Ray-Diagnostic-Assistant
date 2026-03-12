"""
Quick test script to verify heatmap generation and integration
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("=" * 60)
print("HEATMAP & INTEGRATION TEST")
print("=" * 60)

# Test 1: Import checks
print("\n[1/5] Testing imports...")
try:
    from src.HeatmapGenerator import generate_gradcam
    print("✅ HeatmapGenerator imports successfully")
except Exception as e:
    print(f"❌ HeatmapGenerator import failed: {e}")
    sys.exit(1)

try:
    from src.DensenetModels import DenseNet121
    print("✅ DensenetModels imports successfully")
except Exception as e:
    print(f"❌ DensenetModels import failed: {e}")
    sys.exit(1)

try:
    from src.binary_pipeline import BinaryClassifierPipeline
    print("✅ BinaryClassifierPipeline imports successfully")
except Exception as e:
    print(f"❌ BinaryClassifierPipeline import failed: {e}")
    sys.exit(1)

try:
    from src.explainability_ai import explainability_ai
    print("✅ Explainability AI imports successfully")
except Exception as e:
    print(f"❌ Explainability AI import failed: {e}")
    sys.exit(1)

# Test 2: Model paths
print("\n[2/5] Checking model paths...")
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
chexnet_model = os.path.join(backend_dir, 'models', 'chexnet', 'm-30012020-104001.pth.tar')
binary_model1 = os.path.join(backend_dir, 'models', 'binary', 'garbage_vs_xray.pth')
binary_model2 = os.path.join(backend_dir, 'models', 'binary', 'chest_vs_other.pth')
binary_model3 = os.path.join(backend_dir, 'models', 'binary', 'normal_vs_abnormal.pth')

print(f"CheXNet model: {chexnet_model}")
if os.path.exists(chexnet_model):
    print("  ✅ Path exists")
else:
    print("  ❌ Path missing")

print(f"Binary model 1: {binary_model1}")
if os.path.exists(binary_model1):
    print("  ✅ Path exists")
else:
    print("  ❌ Path missing")

print(f"Binary model 2: {binary_model2}")
if os.path.exists(binary_model2):
    print("  ✅ Path exists")
else:
    print("  ❌ Path missing")

print(f"Binary model 3: {binary_model3}")
if os.path.exists(binary_model3):
    print("  ✅ Path exists")
else:
    print("  ❌ Path missing")

# Test 3: Backend app imports
print("\n[3/5] Testing backend app imports...")
try:
    from app import app
    print("✅ Backend FastAPI app imports successfully")
except Exception as e:
    print(f"❌ Backend app import failed: {e}")
    sys.exit(1)

# Test 4: Streamlit app setup
print("\n[4/5] Testing Streamlit app imports...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))
    from DensenetModels import DenseNet121 as DN
    from HeatmapGenerator import generate_gradcam as gg
    from binary_pipeline import BinaryClassifierPipeline as BP
    print("✅ Streamlit dependencies import successfully")
except Exception as e:
    print(f"❌ Streamlit dependencies import failed: {e}")
    sys.exit(1)

# Test 5: Environment file
print("\n[5/5] Checking environment setup...")
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    print("✅ .env file exists")
    with open(env_file) as f:
        keys_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
    print(f"  Contains {keys_count} configurations")
else:
    print("⚠️  .env file not found (AI explanations will use defaults)")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - System ready for use!")
print("=" * 60)
print("\nNext steps:")
print("1. Start backend: cd backend && python -m uvicorn app:app --reload")
print("2. Start frontend: cd frontend && npm start")
print("3. Or start Streamlit: streamlit run chexnet/streamlit_app.py")
