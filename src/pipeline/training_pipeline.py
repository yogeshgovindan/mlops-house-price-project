from src.components.model_trainer import (
    ModelTrainer
)
from src.components.data_transformation import (
    DataTransformation
)
from src.components.data_validation import (
    DataValidation
)
from src.components.data_ingestion import (
    DataIngestion
)
from src.logger import logging

logging.info(
    "Training Pipeline Started"
)


if __name__ == "__main__":

    data_ingestion = (
        DataIngestion()
    )

    train_data, test_data = (
        data_ingestion
        .initiate_data_ingestion()
    )

    data_validation = (
        DataValidation()
    )

    validation_status = (
        data_validation
        .initiate_data_validation(
            train_data,
            test_data
        )
    )

    if validation_status:
        print(
            "Data Validation Passed"
        )

    else:
        raise Exception(
            "Data Validation Failed"
        )

    data_transformation = (
        DataTransformation()
    )

    train_transformed_path, test_transformed_path = (
        data_transformation
        .initiate_data_transformation(
            train_data,
            test_data
        )
    )

    model_trainer = (
        ModelTrainer()
    )

    score = (
        model_trainer
        .initiate_model_training(
            train_transformed_path,
            test_transformed_path
        )
    )

    print(
        f"Final Model Score: {score}"
    )

logging.info(
    "Training Pipeline Completed"
)
