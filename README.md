# Skin Cancer AI

AI-assisted skin lesion screening: a CNN classifier served through a FastAPI backend with a modern web frontend.

> ⚕️ **For education and research only — not a medical device.** This tool cannot diagnose skin cancer. Always consult a dermatologist about concerning skin changes.

## About this project

This app helps you understand what a spot on your skin might be. You upload a close-up photo, and the app compares it against patterns it learned from 10,000 real skin photos, each one labelled by doctors. It then shows you the most likely match out of seven common types of skin spots, including moles, melanoma, and keratoses together with how confident it is.
What sets it apart is that it shows its working: the app highlights the exact area of the photo that influenced its decision, so you can see why it gave that answer instead of just trusting a label.
The final model was tested on 1,503 photos it had never seen before and identified the correct type 76.6% of the time — including catching about 7 in 10 melanomas, the most dangerous type in the set.

Model file needed: the trained model (model.h5, ~109 MB) is too big for this repo. Download it from the [v2.0 release](https://github.com/yannisismname/SkinCancerDetector/releases/tag/v2.0) and place it in the model/ folder, next to classes.json, before running the backend.


## Project structure

```
skin-cancer-ai/
├── backend/      FastAPI service (prediction + Grad-CAM explanations)
├── frontend/     Web UI (drag & drop upload, results, heatmap)
├── training/     Colab notebook that trains the model on HAM10000
└── model/        Trained model files (model.h5 + classes.json) — not in git
```

## Quick start

### 1. Train the model (one time)

1. Open `training/skin_cancer_training.ipynb` in **Google Colab** (Runtime → T4 GPU).
2. Follow the cells: download HAM10000, train, evaluate.
3. Download the exported `model.h5` and `classes.json` into the `model/` folder.

### 2. Run the backend

This project requires Python 3.12. TensorFlow 2.x is not available for Python 3.13 in this setup.

```bash
# Option A: create a local venv with Python 3.12
cd backend
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Or, if you use Conda:

```bash
conda create -n skin-cancer-ai python=3.12
conda activate skin-cancer-ai
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Run the frontend

```bash
cd frontend
python -m http.server 5500
```

Open **http://localhost:5500** — the page checks the backend connection automatically.

## API

| Method | Endpoint        | Description                                          |
|--------|-----------------|------------------------------------------------------|
| GET    | `/`             | Health check                                         |
| GET    | `/diagnose`     | Model info (output shape, class list)                |
| GET    | `/classes`      | Class names                                          |
| POST   | `/predict`      | `{prediction, confidence}` for an uploaded image     |
| POST   | `/predict-all`  | Top prediction + ranked list of all class confidences|
| POST   | `/explain`      | Grad-CAM heatmap overlaid on the image (base64 PNG)  |

## Rebuilding from the old project

This is a clean rebuild of the original `SkiCancerDetector` repo. Fixed along the way:

- `/explain` 500 error (Keras 3 `.output` on unbuilt Sequential) — rewritten Grad-CAM, returns a base64 overlay the frontend can actually display
- `predict` fallback that set a fallback label then crashed indexing anyway
- Uploaded temp files were never deleted — preprocessing is now fully in-memory
- `classes.json` was silently rewritten on disk at startup — now validated, never mutated
- Removed hardcoded personal paths from docs; added `.gitignore` (no more `.bak` files in the repo)
