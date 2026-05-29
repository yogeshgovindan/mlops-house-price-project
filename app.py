from fastapi import FastAPI
import pandas as pd
import pickle
import time
import uuid

from src.schema.prediction_schema import (
    HouseData
)

from src.api.exception_handler import (
    global_exception_handler
)

from src.api.api_logger import (
    logger
)


# ------------------------
# Create FastAPI App
# ------------------------
app = FastAPI()


# ------------------------
# Global Exception Handler
# ------------------------
app.add_exception_handler(
    Exception,
    global_exception_handler
)


# ------------------------
# Global Variables
# ------------------------
model = None
preprocessor = None


# ------------------------
# Startup Event
# ------------------------
@app.on_event("startup")
def load_artifacts():

    global model
    global preprocessor

    try:

        logger.info(
            "API startup initiated"
        )

        # ------------------------
        # Load Preprocessor
        # ------------------------
        with open(
            "artifacts/preprocessor.pkl",
            "rb"
        ) as file:

            preprocessor = pickle.load(
                file
            )

        # ------------------------
        # Load Model
        # ------------------------
        with open(
            "artifacts/model.pkl",
            "rb"
        ) as file:

            model = pickle.load(
                file
            )

        logger.info(
            "Model loaded successfully"
        )

    except Exception as e:

        logger.error(
            f"Startup Error: {e}"
        )


# ------------------------
# Home Route
# ------------------------
@app.get("/")
def home():

    return {
        "status": "success",
        "message":
        "House Price Prediction API Running"
    }


# ------------------------
# Health Check Route
# ------------------------
@app.get("/health")
def health_check():

    if model is not None:

        logger.info(
            "Health check successful"
        )

        return {
            "status":
            "healthy",

            "model_loaded":
            True
        }

    logger.warning(
        "Health check failed"
    )

    return {
        "status":
        "unhealthy",

        "model_loaded":
        False
    }


# ------------------------
# Prediction Route
# ------------------------
@app.post("/predict")
def predict(
    data: HouseData
):

    try:

        # ------------------------
        # Unique Request ID
        # ------------------------
        request_id = str(
            uuid.uuid4()
        )[:8]

        # ------------------------
        # Start Timer
        # ------------------------
        start_time = time.time()

        # ------------------------
        # Check Model Availability
        # ------------------------
        if model is None:

            logger.error(
                f"[Request ID: "
                f"{request_id}] "
                f"Prediction failed: "
                f"Model unavailable"
            )

            return {
                "status":
                "error",

                "request_id":
                request_id,

                "message":
                "Model unavailable"
            }

        logger.info(
            f"[Request ID: "
            f"{request_id}] "
            f"Prediction request received"
        )

        # ------------------------
        # Convert Input
        # ------------------------
        input_df = pd.DataFrame(
            [data.model_dump()]
        )

        # ------------------------
        # Apply Preprocessing
        # ------------------------
        transformed_data = (
            preprocessor.transform(
                input_df
            )
        )

        transformed_data = pd.DataFrame(
            transformed_data
        )

        # ------------------------
        # Prediction
        # ------------------------
        prediction = model.predict(
            transformed_data
        )

        logger.info(
            f"[Request ID: "
            f"{request_id}] "
            f"Prediction output: "
            f"{prediction[0]}"
        )

        # ------------------------
        # Response Time
        # ------------------------
        response_time = (
            time.time()
            - start_time
        )

        logger.info(
            f"[Request ID: "
            f"{request_id}] "
            f"Response time: "
            f"{response_time:.2f} sec"
        )

        logger.info(
            f"[Request ID: "
            f"{request_id}] "
            f"Prediction successful"
        )

        return {
            "status":
            "success",

            "request_id":
            request_id,

            "predicted_house_price":
            float(
                prediction[0]
            )
        }

    except Exception as e:

        logger.error(
            f"Prediction Error: {e}"
        )

        raise e