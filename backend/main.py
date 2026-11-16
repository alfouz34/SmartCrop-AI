from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import recommend_crops, load_model, MODEL_PATH
from src.preprocessing import FEATURES

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as exc:
        print(f"Warning on startup: model could not be preloaded: {exc}")
    yield

app = FastAPI(
    title="SmartCrop AI API",
    description="Crop recommendations from soil nutrients and climate inputs.",
    version="1.0.0",
    lifespan=lifespan,
)

class CropInput(BaseModel):
    N: float = Field(..., ge=0, le=300, description="Nitrogen level in soil (kg/ha)")
    P: float = Field(..., ge=0, le=300, description="Phosphorus level in soil (kg/ha)")
    K: float = Field(..., ge=0, le=400, description="Potassium level in soil (kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in Celsius (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH value (0 to 14)")
    rainfall: float = Field(..., ge=0, le=1000, description="Rainfall in millimeters (mm)")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "SmartCrop AI",
        "model_loaded": MODEL_PATH.exists()
    }

@app.post("/predict")
def predict(data: CropInput):
    values = data.model_dump()
    try:
        recommendations = recommend_crops(values)
        return {
            "status": "success",
            "recommendations": recommendations,
            "note": "Model scores are model prediction scores based on predict_proba(), which averages the class probability estimates produced by the individual decision trees and has not been calibrated. They are not guaranteed real-world probabilities of crop success. Use as agricultural decision support alongside local expertise."
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
