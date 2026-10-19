"""Skin Cancer AI backend — FastAPI service.

Run from the backend/ folder:
    uvicorn main:app --host 0.0.0.0 --port 5000 --reload

The trained model lives in ../model/ (model.h5 + classes.json).
"""
import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_loader import ModelLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Skin Cancer AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

ml = ModelLoader()
logger.info("Model loaded successfully")


@app.get("/")
def root():
    return {"message": "Skin Cancer AI Backend is running"}


@app.get("/diagnose")
def diagnose():
    return {
        "model_output_shape": list(ml.model.output_shape),
        "class_count": len(ml.class_names),
        "classes": ml.class_names,
    }


@app.get("/classes")
def classes():
    return {"classes": ml.class_names}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img_bytes = await image.read()
    try:
        label, _, confidence = ml.predict(img_bytes)
        return JSONResponse({"prediction": label, "confidence": confidence})
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": "Prediction failed", "detail": str(e)}, status_code=500)


@app.post("/predict-all")
async def predict_all(image: UploadFile = File(...)):
    """Top prediction plus the full ranked list of class confidences."""
    img_bytes = await image.read()
    try:
        ranked = ml.predict_all(img_bytes)
        return JSONResponse({"prediction": ranked[0]["label"],
                             "confidence": ranked[0]["confidence"],
                             "all": ranked})
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse({"error": "Prediction failed", "detail": str(e)}, status_code=500)


@app.post("/explain")
async def explain(image: UploadFile = File(...)):
    """Grad-CAM heatmap overlaid on the input image, as a base64 data URI."""
    img_bytes = await image.read()
    try:
        heatmap_uri = ml.explain(img_bytes)
        return JSONResponse({"heatmap": heatmap_uri})
    except Exception as e:
        logger.exception("Explanation failed")
        return JSONResponse({"error": "Explanation failed", "detail": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
