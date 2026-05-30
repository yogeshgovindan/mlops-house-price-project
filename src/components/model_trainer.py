import sys
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object


class ModelTrainer:

    def __init__(self):
        self.model_path = "artifacts/model.pkl"

        # ✅ FIX 1: FORCE SAFE CROSS-PLATFORM PATH (IMPORTANT)
        self.mlflow_uri = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(self.mlflow_uri)

    def initiate_model_training(self, train_path, test_path):

        try:
            logging.info("Entered Model Training")

            # ------------------------
            # Load data
            # ------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "median_house_value"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # ------------------------
            # Models
            # ------------------------
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42)
            }

            logging.info("Model evaluation started")

            model_report = evaluate_models(
                X_train, y_train,
                X_test, y_test,
                models
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Score: {best_model_score}")

            # ------------------------
            # 🔥 MLflow FIX START (IMPORTANT)
            # ------------------------

            # Force experiment creation safely
            mlflow.set_experiment("house_price_prediction")

            with mlflow.start_run(run_name=best_model_name):

                mlflow.log_param("model", best_model_name)
                mlflow.log_metric("r2_score", best_model_score)

                # Log model safely
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model"
                )

                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"

                mlflow.register_model(
                    model_uri=model_uri,
                    name="house_price_model"
                )

            logging.info("MLflow tracking completed")

            # ------------------------
            # Save model locally
            # ------------------------
            save_object(
                file_path=self.model_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
