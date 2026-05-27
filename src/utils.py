import os
import sys
import pickle

from sklearn.metrics import (
    r2_score
)

from src.exception import (
    CustomException
)


def save_object(
    file_path,
    obj
):

    try:

        dir_path = (
            os.path.dirname(
                file_path
            )
        )

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        with open(
            file_path,
            "wb"
        ) as file_obj:

            pickle.dump(
                obj,
                file_obj
            )

    except Exception as e:

        raise CustomException(
            e,
            sys
        )


def load_object(
    file_path
):

    try:

        with open(
            file_path,
            "rb"
        ) as file_obj:

            return pickle.load(
                file_obj
            )

    except Exception as e:

        raise CustomException(
            e,
            sys
        )


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models
):

    try:

        report = {}

        for (
            model_name,
            model
        ) in models.items():

            model.fit(
                X_train,
                y_train
            )

            predictions = (
                model.predict(
                    X_test
                )
            )

            score = (
                r2_score(
                    y_test,
                    predictions
                )
            )

            report[
                model_name
            ] = score

        return report

    except Exception as e:

        raise CustomException(
            e,
            sys
        )
