import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ----------------------------
# PATHS
# ----------------------------
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"


# ----------------------------
# LOAD MODEL IMMEDIATELY
# (pytest-safe fix)
# ----------------------------
model = None
preprocessor = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)


# ----------------------------
# INPUT SCHEMA
# ----------------------------
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


# ----------------------------
# HOME
# ----------------------------
@app.get("/")
def home():
    return {
        "status": "success"
    }


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ----------------------------
# PREDICT
# ----------------------------
@app.post("/predict")
def predict(data: HouseFeatures):

    if model is None:
        return {
            "error": "Model not loaded"
        }

    if preprocessor is None:
        return {
            "error": "Preprocessor not loaded"
        }

    input_df = pd.DataFrame([
        data.model_dump()
    ])

    transformed_data = preprocessor.transform(
        input_df
    )

    prediction = model.predict(
        transformed_data
    )

    return {
        "predicted_house_price":
        float(prediction[0])
    }
