from fastapi.testclient import (
    TestClient
)

from app import app


# ------------------------
# Create Test Client
# ------------------------
client = TestClient(app)


# ------------------------
# Test Home Endpoint
# ------------------------
def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert (
        data["status"]
        == "success"
    )


# ------------------------
# Test Health Endpoint
# ------------------------
def test_health():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        "status"
        in data
    )

    assert (
        "model_loaded"
        in data
    )


# ------------------------
# Test Prediction Endpoint
# ------------------------
def test_prediction():

    payload = {

        "longitude":
        -122.23,

        "latitude":
        37.88,

        "housing_median_age":
        41,

        "total_rooms":
        880,

        "total_bedrooms":
        129,

        "population":
        322,

        "households":
        126,

        "median_income":
        8.32,

        "ocean_proximity":
        "NEAR BAY"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        "predicted_house_price"
        in data
        or
        "message"
        in data
    )


# ------------------------
# Test Invalid Input
# ------------------------
def test_invalid_input():

    payload = {

        "longitude":
        "hello",

        "latitude":
        37.88
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert (
        response.status_code
        == 422
    )
