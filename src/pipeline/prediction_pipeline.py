import pickle
import pandas as pd


class PredictionPipeline:

    def predict(
        self,
        input_data
    ):

        with open(
            "artifacts/preprocessor.pkl",
            "rb"
        ) as file:

            preprocessor = (
                pickle.load(file)
            )

        with open(
            "artifacts/model.pkl",
            "rb"
        ) as file:

            model = (
                pickle.load(file)
            )

        transformed_data = (
            preprocessor.transform(
                input_data
            )
        )

        prediction = (
            model.predict(
                transformed_data
            )
        )

        return prediction[0]
