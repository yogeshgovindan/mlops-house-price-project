from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = (
    "house_price_model"
)

version = 2

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)

print(
    f"Model Version {version} promoted to Production"
)
