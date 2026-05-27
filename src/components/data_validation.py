import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import (
    CustomException
)


class DataValidation:

    def initiate_data_validation(
        self,
        train_path,
        test_path
    ):

        logging.info(
            "Entered data validation "
            "method"
        )

        try:

            train_df = pd.read_csv(
                train_path
            )

            logging.info(
                "Train dataset loaded"
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Test dataset loaded"
            )

            required_columns = [

                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "median_house_value",
                "ocean_proximity"
            ]

            validation_status = True

            for column in (
                required_columns
            ):

                if (
                    column
                    not in train_df.columns
                ):

                    validation_status = (
                        False
                    )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            with open(
                "artifacts/validation_status.txt",
                "w"
            ) as file:

                file.write(
                    str(
                        validation_status
                    )
                )

            if validation_status:

                logging.info(
                    "Data validation "
                    "passed"
                )

            else:

                logging.info(
                    "Data validation "
                    "failed"
                )

            logging.info(
                "Data validation "
                "completed"
            )

            return validation_status

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
