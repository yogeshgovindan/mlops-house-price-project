import os
import sys
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.logger import logging
from src.exception import (
    CustomException
)


class DataIngestionConfig:

    raw_data_path = (
        "artifacts/raw.csv"
    )

    train_data_path = (
        "artifacts/train.csv"
    )

    test_data_path = (
        "artifacts/test.csv"
    )


class DataIngestion:

    def __init__(self):

        self.ingestion_config = (
            DataIngestionConfig()
        )

    def initiate_data_ingestion(
        self
    ):

        logging.info(
            "Entered data ingestion "
            "method"
        )

        try:

            df = pd.read_csv(
                "data/housing.csv"
            )

            logging.info(
                "Dataset loaded "
                "as dataframe"
            )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config
                .raw_data_path,
                index=False
            )

            logging.info(
                "Raw data saved"
            )

            train_set, test_set = (
                train_test_split(
                    df,
                    test_size=0.2,
                    random_state=42
                )
            )

            logging.info(
                "Train test split "
                "completed"
            )

            train_set.to_csv(
                self.ingestion_config
                .train_data_path,
                index=False
            )

            test_set.to_csv(
                self.ingestion_config
                .test_data_path,
                index=False
            )

            logging.info(
                "Train and test "
                "files saved"
            )

            logging.info(
                "Data ingestion "
                "completed"
            )

            return (
                self.ingestion_config
                .train_data_path,

                self.ingestion_config
                .test_data_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
