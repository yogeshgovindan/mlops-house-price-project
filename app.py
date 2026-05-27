from fastapi import FastAPI, HTTPException
import pandas as pd
import logging

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.prediction_api_schema import HouseData

app = FastAPI()

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------- Health Check ----------------
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is live"
    }


# ---------------- Prediction Endpoint ----------------
@app.post("/predict")
def predict(data: HouseData):

    try:
        logging.info("Prediction request received")

        # convert request → DataFrame
        input_df = pd.DataFrame([data.dict()])

        # load pipeline
        pipeline = PredictionPipeline()

        # prediction
        prediction = pipeline.predict(input_df)

        logging.info("Prediction successful")

        return {
            "predicted_house_price": float(prediction)
        }

    except Exception as e:

        logging.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )
