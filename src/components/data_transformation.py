import os
import sys
import pickle
import pandas as pd

from sklearn.impute import (
    SimpleImputer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import (
    ColumnTransformer
)

from src.logger import logging

from src.exception import (
    CustomException
)


class DataTransformationConfig:

    transformed_train_path = (
        "artifacts/train_transformed.csv"
    )

    transformed_test_path = (
        "artifacts/test_transformed.csv"
    )

    preprocessor_path = (
        "artifacts/preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.config = (
            DataTransformationConfig()
        )

    def get_data_transformer(
        self
    ):

        numerical_columns = [

            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]

        categorical_columns = [
            "ocean_proximity"
        ]

        numerical_pipeline = (
            Pipeline(
                steps=[

                    (
                        "imputer",

                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",

                        StandardScaler()
                    )
                ]
            )
        )

        categorical_pipeline = (
            Pipeline(
                steps=[

                    (
                        "imputer",

                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",

                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )
        )

        preprocessor = (
            ColumnTransformer(
                [

                    (
                        "numerical_pipeline",

                        numerical_pipeline,

                        numerical_columns
                    ),

                    (
                        "categorical_pipeline",

                        categorical_pipeline,

                        categorical_columns
                    )
                ]
            )
        )

        return preprocessor

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        logging.info(
            "Entered data "
            "transformation method"
        )

        try:

            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Train and test "
                "datasets loaded"
            )

            preprocessing_obj = (
                self
                .get_data_transformer()
            )

            logging.info(
                "Preprocessor "
                "object created"
            )

            target_column = (
                "median_house_value"
            )

            # training split
            X_train = (
                train_df.drop(
                    columns=[
                        target_column
                    ]
                )
            )

            y_train = (
                train_df[
                    target_column
                ]
            )

            # testing split
            X_test = (
                test_df.drop(
                    columns=[
                        target_column
                    ]
                )
            )

            y_test = (
                test_df[
                    target_column
                ]
            )

            X_train_transformed = (
                preprocessing_obj
                .fit_transform(
                    X_train
                )
            )

            X_test_transformed = (
                preprocessing_obj
                .transform(
                    X_test
                )
            )

            logging.info(
                "Data transformation "
                "completed"
            )

            train_transformed_df = (
                pd.DataFrame(
                    X_train_transformed
                )
            )

            test_transformed_df = (
                pd.DataFrame(
                    X_test_transformed
                )
            )

            train_transformed_df[
                target_column
            ] = (
                y_train.values
            )

            test_transformed_df[
                target_column
            ] = (
                y_test.values
            )

            train_transformed_df.to_csv(
                self.config
                .transformed_train_path,
                index=False
            )

            test_transformed_df.to_csv(
                self.config
                .transformed_test_path,
                index=False
            )

            logging.info(
                "Transformed files "
                "saved"
            )

            with open(
                self.config
                .preprocessor_path,
                "wb"
            ) as file:

                pickle.dump(
                    preprocessing_obj,
                    file
                )

            logging.info(
                "Preprocessor saved"
            )

            return (

                self.config
                .transformed_train_path,

                self.config
                .transformed_test_path
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )
