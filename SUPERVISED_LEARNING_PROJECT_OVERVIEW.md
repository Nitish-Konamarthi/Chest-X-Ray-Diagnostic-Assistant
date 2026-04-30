# CheXNet-Proto: Supervised Learning and Project Overview

## 1. Is this a supervised learning project?
Yes — your project is a supervised learning system.

### Why?
- Every model is trained with images paired with known labels.
- The code reads image paths plus disease labels from dataset files like `dataset/train_1.txt`, `dataset/val_1.txt`, and `dataset/test_1.txt`.
- The training process uses a loss function (`BCELoss`) to compare model outputs against ground-truth labels.
- The model weights are updated during training to reduce that loss.

So the model learns from examples where the correct answer is already known.

## 2. Which models are supervised?
All four image models in your project are supervised:

1. `MobileNetV2` in `backend/src/binary_pipeline.py`
   - Task: Garbage vs X-ray
   - Output: 2 classes

2. `ResNet18` in `backend/src/binary_pipeline.py`
   - Task: Chest vs Other X-ray
   - Output: 2 classes

3. `ResNet18` in `backend/src/binary_pipeline.py`
   - Task: Normal vs Abnormal
   - Output: 2 classes

4. `DenseNet121` in `backend/src/DensenetModels.py`
   - Task: 14-pathology classification (CheXNet-like)
   - Output: 14 sigmoid probabilities

Each of these models is trained using labeled data and supervised loss.

## 3. How training works in this project
### Training script and entry point
The training code is located in the `chexnet/` folder:
- `chexnet/Main.py`
- `chexnet/ChexnetTrainer.py`
- `chexnet/DatasetGenerator.py`

The `Main.py` script shows the training flow:
- `pathDirData = './database'`
- `pathFileTrain = './dataset/train_1.txt'`
- `pathFileVal = './dataset/val_1.txt'`
- `pathFileTest = './dataset/test_1.txt'`
- Architecture: `DENSE-NET-121`
- Pretrained on ImageNet: `nnIsTrained = True`
- Number of classes: `14`
- Batch size: `16`
- Max epochs: `100`
- Image crop size: `224`

### Data format
The dataset files contain lines like:

```
images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Each line includes:
- image path
- label vector for 14 disease outputs

Those label vectors are the supervised targets.

### Dataset loader
`chexnet/DatasetGenerator.py` does this:
- reads each line from dataset file
- builds `imagePath`
- converts each label item to an integer
- loads the image with PIL
- converts it to RGB
- applies transforms
- returns `(imageTensor, labelTensor)`

### Training loop
`chexnet/ChexnetTrainer.py` trains with:
- model = `DenseNet121(...)`
- optimizer = `Adam` with learning rate `1e-4`
- loss = `torch.nn.BCELoss(size_average=True)`
- scheduler = `ReduceLROnPlateau`

Training steps per batch:
1. load image batch and label batch
2. forward pass through the model
3. compute binary cross-entropy loss between predictions and labels
4. backward pass
5. optimizer step to update weights

That is the standard supervised training process.

## 4. What preprocessing is applied
### During training and inference
For the binary pipeline (`backend/src/binary_pipeline.py`):
- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`

For CheXNet inference in `backend/app.py`:
- `Resize(256)`
- `TenCrop(224)`
- convert each crop to tensor
- normalize with ImageNet means/stds

### Why this preprocessing?
- Resizing makes the image a consistent size.
- Cropping selects the central region where the chest is expected.
- Normalization uses ImageNet mean/std because the model begins from ImageNet pretrained weights.
- For CheXNet, `TenCrop` creates 10 views for more stable predictions.

## 5. What the model does from input to output
### Input
The user uploads a chest X-ray image.

### Step 1: Binary pipeline validation
The backend first runs the 3-stage pipeline in `backend/src/binary_pipeline.py`.

The pipeline checks:
- Model 1: Is this a medical X-ray or garbage?
- Model 2: If it is an X-ray, is it a chest X-ray?
- Model 3: If it is a chest X-ray, is it normal or abnormal?

If any check fails, the pipeline stops and returns a message.
If the image is valid and abnormal, the app continues to CheXNet analysis.

### Step 2: CheXNet pathology analysis
In `backend/app.py`, the app loads the main pathology model:
- `DenseNet121(14, True)`
- uses weights from `backend/models/chexnet/m-30012020-104001.pth.tar`

The model produces 14 probability scores, one for each pathology.

### Step 3: Clinical summary and thresholds
The backend applies clinical thresholds such as:
- `Atelectasis: 0.45`
- `Cardiomegaly: 0.55`
- `Pneumonia: 0.40`
- `Hernia: 0.70`

It classifies the findings into `HIGH`, `MEDIUM`, `LOW`, or `MINIMAL` confidence.
It generates a summary like:
- `NORMAL`
- `ABNORMAL`
- `BORDERLINE`

### Step 4: Explainability
The system also generates:
- Grad-CAM heatmaps via `backend/src/HeatmapGenerator.py`
- AI explanation text using `backend/src/explainability_ai.py`

### Step 5: Nearby doctor recommendation
The backend can call `geoapify_finder` in `backend/src/geoapify_service.py` to suggest specialists.
This is based on the detected pathology and user location.

## 6. Project architecture summary
### Frontend
- React app in `frontend/src`
- file upload and display
- shows heatmap, pathology results, and doctor suggestions

### Backend
- FastAPI app in `backend/app.py`
- loads CheXNet and binary models
- exposes `/analyze` endpoint for image upload
- returns analysis, heatmap, explanations, and doctor data

### Models
- `backend/src/DensenetModels.py`: DenseNet variants for CheXNet
- `backend/src/binary_pipeline.py`: 3x binary validation models
- `backend/src/HeatmapGenerator.py`: generates Grad-CAM++ overlays
- `backend/src/explainability_ai.py`: formats Gemini AI prompts and responses
- `backend/src/geoapify_service.py`: doctor search logic

### Training support
- `chexnet/ChexnetTrainer.py`: supervised training loop for DenseNet
- `chexnet/DatasetGenerator.py`: dataset loader for image + label pairs
- `chexnet/Main.py`: training and test runner template

## 7. Supervised training -> testing -> ready state
### Training phase
- The model is trained on labeled chest X-rays.
- Each image is matched with known disease labels.
- The network learns to map image patterns to disease probabilities.
- Training updates weights with gradient descent.

### Testing phase
- The model is evaluated on a separate test set from `dataset/test_1.txt`.
- It computes validation loss and AUROC for each disease.
- This verifies how well the model generalizes.

### Ready state
- The final weights are saved as `m-*.pth.tar`.
- The backend loads the saved weights at startup.
- The system is ready to receive new X-ray uploads and make predictions.

## 8. Important practical points
- The model does not use age, gender, or other patient metadata.
- It only uses the image pixels.
- It is a supervised image classification system.
- The main pathology model is CheXNet-style multi-label classification.
- The binary pipeline is a custom validation layer built around the main model.

## 9. How to run in your repo
### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### Training (optional)
```bash
cd chexnet
python Main.py
```

## 10. Best short answer for supervised learning
- The project is supervised because it uses labeled X-ray images.
- The model learns from exact image-label pairs.
- All four image models in this repo are supervised.
- The training pipeline is standard: preprocess → forward pass → loss → backward pass → update.

---

This `.md` file explains the entire project flow from training to ready state, including preprocessing, the supervised learning approach, and how inference works in your CheXNet-inspired application.
