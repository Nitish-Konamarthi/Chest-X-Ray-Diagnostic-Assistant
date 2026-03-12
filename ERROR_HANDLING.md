"""
SYSTEM ARCHITECTURE & ERROR HANDLING
====================================

This document explains the system architecture and how errors are handled.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                          │
├──────────────────┬─────────────────────────┬────────────────┤
│  React Frontend  │  Streamlit Web App     │  Direct API   │
│  (Port 3000)     │  (Port 8501)           │  (Port 8000)  │
└──────────────────┴───────┬─────────────────┴────────────────┘
                           │ HTTP/REST
                           ▼
        ┌──────────────────────────────────┐
        │   FastAPI Backend (Port 8000)    │
        │  ├─ /analyze endpoint            │
        │  ├─ /health endpoint             │
        │  └─ CORS middleware (React+Streamlit)
        └──────┬───────────────────────────┘
               │
        ┌──────┴───────────────────────────┐
        │                                  │
        ▼                                  ▼
┌───────────────────────┐       ┌──────────────────────┐
│  Image Validation     │       │  Model Inference     │
│ (Binary Pipeline)     │       │ (CheXNet DenseNet121)│
├───────────────────────┤       ├──────────────────────┤
│ Model1: Garbage/Xray  │       │ 14 Pathologies      │
│ Model2: Chest/Other   │       │ Clinical thresholds │
│ Model3: Normal/Abnorm │       │ AI Predictions      │
└───────────────────────┘       └──────────────────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
              ┌────────────────────┐
              │  Post-Processing   │
              ├────────────────────┤
              │ Grad-CAM++ Heatmap │
              │ AI Explanations    │
              │ Clinical Summary   │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │  JSON Response     │
              │ + Base64 Heatmap   │
              │ + LLM Explanation  │
              └────────────────────┘
```

## Error Handling Strategy

### Level 1: Input Validation
- File type check (PNG, JPG, TIFF)
- File size limit
- Image resolution limits

### Level 2: Binary Pipeline Errors
- Model loading failures → Gracefully skip
- Validation failures → Return invalid_for_analysis: false
- Device errors → Fallback from CUDA to CPU

### Level 3: Main Model Errors
- Model loading failures → Application startup error
- Inference failures → Caught with try-except

### Level 4: Post-Processing Errors
- Heatmap generation failures → heatmap_b64: null
- AI explanation failures → Use default explanation
- API timeouts → Fallback to next API

## Error Recovery

### Automatic Fallbacks

1. **GPU to CPU**: If CUDA fails, automatically use CPU
2. **Gemini to ChatGPT**: If Gemini API fails, try ChatGPT
3. **ChatGPT to Default**: If all APIs fail, use default explanation
4. **Heatmap Generation**: If fails, return null (image still displayed)

### Manual Interventions

1. **Model not found**: Clear error message with path
2. **API key missing**: Warning shown, system continues
3. **CUDA out of memory**: Error caught, suggestion given
4. **File upload error**: Clear error message shown to user

## Component Error Handling

### Backend (FastAPI)
```python
# Endpoint error handling
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile):
    # 1. File validation
    if not valid_file_type:
        raise HTTPException(400, "Invalid file type")
    
    # 2. Binary pipeline
    try:
        validation_result = binary_pipeline.validate(image)
    except:
        return invalid_response
    
    # 3. Main model inference
    try:
        results = predict_chexnet(image)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    
    # 4. Heatmap generation (non-critical)
    try:
        heatmap = generate_heatmap_b64(image, inp)
    except:
        heatmap = None  # Graceful degradation
    
    # 5. AI explanation (non-critical)
    try:
        explanation = explainability_ai.generate_explanation(...)
    except:
        explanation = default_explanation
    
    return response  # All together
```

### React Frontend
```javascript
// Error handling in upload
try {
  const response = await fetch('/analyze', {...});
  if (!response.ok) {
    setError(`HTTP ${response.status}: ${response.statusText}`);
  }
  const data = await response.json();
  setResults(data);
} catch (err) {
  setError(`Connection error: ${err.message}`);
  // Shows: "Network error", "CORS error", etc.
}

// Displaying results safely
if (results?.heatmap_b64) {
  // Display heatmap
} else {
  // Show without heatmap (graceful degradation)
}

if (results?.ai_explanation) {
  // Display AI explanation
} else {
  // Show clinical summary only
}
```

### Streamlit App
```python
# Error handling in model loading
try:
    with st.spinner("Loading model..."):
        model, device = load_model(model_path)
    st.success("✅ Model loaded")
except FileNotFoundError:
    st.error("❌ Model file not found")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load: {e}")
    st.stop()

# Error handling in analysis
try:
    results = predict_pil(image, model, device)
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# Non-critical errors
try:
    heatmap = generate_gradcam_visualization(...)
except Exception as e:
    st.warning(f"⚠️ Could not generate heatmap: {e}")
    # Continue without heatmap
```

## Logging & Debugging

### Backend Logging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# In app.py
logger = logging.getLogger(__name__)
logger.info("Model loaded successfully")
logger.warning("Heatmap generation failed")
logger.error("Critical error occurred")
```

### Frontend Debugging
```javascript
// Browser console logging
console.log("Uploading file...");
console.error("API Error:", error);

// React DevTools
// Check state changes during analysis
```

### Streamlit Debugging
```python
# Add debugging output
if debug_mode:
    st.write("Model path:", model_path)
    st.write("Image shape:", image.shape)
    st.write("Prediction shape:", results.shape)
```

## Common Error Scenarios

### Scenario 1: CUDA Out of Memory
Error Message: "CUDA out of memory"
Root Cause: Large batch size or GPU memory full
Solution:
1. Reduce image resolution
2. Use CPU instead: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
3. Restart backend to clear memory

### Scenario 2: Model File Missing
Error Message: "Model not found at {PATH}"
Root Cause: Incorrect path or file deleted
Solution:
1. Run: ls backend/models/chexnet/
2. Check model file exists
3. Restart backend

### Scenario 3: Heatmap Generation Fails
Error Message: "Grad-CAM++ generation failed"
Root Cause: Hook registration issues, model in wrong mode
Solution:
1. System will show image without heatmap (graceful)
2. Check backend logs for details
3. Try restarting backend

### Scenario 4: API Connection Failed
Error Message: "Failed to fetch from localhost:8000"
Root Cause: Backend not running, firewall, or CORS issue
Solution:
1. Check backend is running: curl http://localhost:8000/health
2. Check CORS settings in backend/app.py
3. Open http://localhost:3000 (not file://)

### Scenario 5: Binary Validation Failed
Error Message: "Image validation failed"
Root Cause: Not a chest X-ray or invalid image
Solution:
1. Ensure valid medical imaging file
2. Try different image format
3. Check binary pipeline is working: curl /health

### Scenario 6: No AI Explanations
Error Message: Missing AI explanation in response
Root Cause: API keys not configured or API unavailable
Solution:
1. This is NOT an error - system works without explanations
2. Optional: Add API keys to .env
3. Optional: Configure both Gemini and OpenAI for fallback

### Scenario 7: Frontend Stuck on Loading
Error Message: "Loading..." spinner stuck
Root Cause: Backend timeout, large file, or network issue
Solution:
1. Check browser console for errors
2. Verify backend is responding: curl /health
3. Try smaller image file
4. Check network tab for failed requests

## Performance Under Load

### Single Request (Expected)
- Binary Pipeline: 50-100ms
- Model Inference: 150-200ms
- Heatmap Generation: 50-100ms
- AI Explanation: 500-3000ms (API dependent)
- Total: 750ms - 3300ms

### Multiple Concurrent Requests
- Single instance: Processes sequentially
- Expected: ~1-2s per request with queue

## Error Monitoring

### Health Check Endpoint
```bash
curl http://localhost:8000/health
Response: {"status": "healthy", "models_loaded": true}
```

### Backend Logs
```bash
tail -f  # Check for errors
grep "ERROR" logs.txt  # Find all errors
```

### Frontend Console
```javascript
// Check browser console for JavaScript errors
// Look for failed fetch requests in Network tab
```

## Recovery Procedures

### Restart Backend
```bash
# Kill existing process
pkill -f "uvicorn"
# Start fresh
cd backend && python -m uvicorn app:app --reload
```

### Reset Frontend Cache
```bash
# Clear React cache
rm -rf node_modules/.cache
npm start
```

### Reset Streamlit Cache
```bash
# Clear Streamlit cache
streamlit cache clear
streamlit run chexnet/streamlit_app.py
```

### Full System Reset
```bash
# Kill all services
pkill -f "uvicorn"
pkill -f "npm"
pkill -f "streamlit"

# Clear caches
rm -rf frontend/node_modules/.cache
rm -rf ~/.streamlit/cache

# Restart all services
# Follow QUICKSTART.md
```

## Testing Error Scenarios

### Test Invalid File Upload
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@invalid.txt"
# Expected: HTTP 400 error
```

### Test Missing Model
```bash
# Remove model file then restart backend
# Expected: Clear error message on startup
```

### Test API Timeout
```bash
# Disable internet then try analysis
# Expected: Graceful fallback to default explanation
```

## Monitoring & Alerts

### Recommended Monitoring
1. Check health endpoint every 30s
2. Monitor GPU memory usage
3. Track average response time
4. Count failed requests

### Alert Conditions
- Health check fails
- Average response > 5s
- GPU memory > 90%
- Failed requests > 5%

## Support Resources

- Check logs: backend logs show detailed errors
- Browser console: Frontend errors and network issues
- Streamlit terminal: Console output
- test_integration.py: System diagnostics

---
Last Updated: 2024
Version: 1.0
"""