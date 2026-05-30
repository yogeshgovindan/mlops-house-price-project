from fastapi import FastAPI
import joblib
import os

app = FastAPI()

MODEL_PATH = "artifacts/model.pkl"
model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"status": "success", "message": "House Price Prediction API Running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([data["features"]])
    return {"prediction": prediction.tolist()}
