import pandas as pd
import mlflow.pyfunc
import pickle


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
# Load Production Model
# ------------------------
model = mlflow.pyfunc.load_model(
    model_uri="models:/house_price_model/Production"
)

# ------------------------
# Sample Input
# ------------------------
sample_data = pd.DataFrame([
    {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.32,
        "ocean_proximity": "NEAR BAY"
    }
])

# ------------------------
# Transform Input
# ------------------------
transformed_data = (
    preprocessor.transform(
        sample_data
    )
)

# ------------------------
# Prediction
# ------------------------
prediction = model.predict(
    transformed_data
)

print(
    "Predicted House Price:",
    prediction[0]
)
