class ModelTrainer:

    model_path = "artifacts/model.pkl"

    def initiate_model_training(
        self,
        train_path,
        test_path
    ):

        logging.info(
            "Entered model training"
        )

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Train and test data loaded"
            )

            target_column = "median_house_value"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor()
            }

            # 🚀 USE UTILITY FUNCTION
            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models
            )

            logging.info(
                f"Model evaluation report: {model_report}"
            )

            # get best model
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}"
            )

            # SAVE MODEL USING UTILITY
            save_object(
                file_path=self.model_path,
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            logging.info(
                "Model training completed"
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
