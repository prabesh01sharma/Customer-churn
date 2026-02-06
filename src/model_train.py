import sys
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelTrainer:
    def __init__(
        self,
        transformed_dir: str | Path = PROJECT_ROOT / "transformed_data",
        models_dir: str | Path = PROJECT_ROOT / "models",
        model_filename: str = "logistic_regression.pkl",
        target_col: str = "churn",
        random_state: int = 42,
    ) -> None:
        self.transformed_dir = Path(transformed_dir)
        self.models_dir = Path(models_dir)
        self.model_filename = model_filename
        self.target_col = target_col.lower()
        self.random_state = random_state

        self.X_train_path = self.transformed_dir / "X_train.csv"
        self.X_test_path = self.transformed_dir / "X_test.csv"
        self.y_train_path = self.transformed_dir / "y_train.csv"
        self.y_test_path = self.transformed_dir / "y_test.csv"

    def _load_transformed_data(self):
        try:
            logging.info("Loading transformed data")
            for p in [self.X_train_path, self.X_test_path, self.y_train_path, self.y_test_path]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing transformed file: {p}")

            X_train = pd.read_csv(self.X_train_path)
            X_test = pd.read_csv(self.X_test_path)

            y_train = pd.read_csv(self.y_train_path)[self.target_col]
            y_test = pd.read_csv(self.y_test_path)[self.target_col]

            logging.info("Transformed data loaded successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_save(self):
        logging.info("Entered the train_and_save method")
        try:
            X_train, X_test, y_train, y_test = self._load_transformed_data()

            # Logistic Regression (baseline)
            logging.info("Initializing LogisticRegression model with balanced class weights")
            model = LogisticRegression(max_iter=3000, random_state=self.random_state, class_weight='balanced')
            
            logging.info("Training the model")
            model.fit(X_train, y_train)

            logging.info("Predicting on test data")
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logging.info("=== Model Training Summary ===")
            logging.info("Model: LogisticRegression")
            logging.info(f"Accuracy: {acc}")
            logging.info(f"Classification Report:\n{report}")
            logging.info("==============================")

            # Save model
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / self.model_filename

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logging.info(f"Saved model to: {model_path.resolve()}")
            return model
        
        except Exception as e:
            raise CustomException(e, sys)


def main() -> None:
    try:
        logging.info("Starting model training execution")
        trainer = ModelTrainer()
        trainer.train_and_save()
    except Exception as e:
        logging.error("Exception occured in model training main execution")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
