import sys
from pathlib import Path
import pickle
import pandas as pd

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelPredictor:
    def __init__(
        self,
        model_path: str | Path = PROJECT_ROOT / "models" / "logistic_regression.pkl",
        preprocessor_path: str | Path = PROJECT_ROOT / "transformed_data" / "preprocessor.pkl",
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor not found: {self.preprocessor_path}")

            logging.info(f"Loading model from {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            logging.info(f"Loading preprocessor from {self.preprocessor_path}")
            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_one(self, input_dict: dict) -> dict:
        """
        input_dict must contain all feature columns used during training
        (same as validated dataset minus target + customerid)
        """
        logging.info("Predicting for single input")
        try:
            X = pd.DataFrame([input_dict])

            # Make columns lowercase to match training pipeline (if you used lowercase everywhere)
            X.columns = [c.strip().lower() for c in X.columns]

            X_t = self.preprocessor.transform(X)
            proba = float(self.model.predict_proba(X_t)[0, 1])
            pred = int(self.model.predict(X_t)[0])

            result = {
                "prediction": pred,                 # 0 = No churn, 1 = Churn
                "probability_churn": proba
            }
            logging.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def predict_csv(self, csv_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
        logging.info(f"Batch prediction for CSV: {csv_path}")
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]

            X_t = self.preprocessor.transform(df)
            proba = self.model.predict_proba(X_t)[:, 1]
            pred = self.model.predict(X_t)

            out = df.copy()
            out["prediction"] = pred
            out["probability_churn"] = proba

            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(output_path, index=False)
                logging.info(f"Saved batch predictions to {output_path}")

            return out
        except Exception as e:
            raise CustomException(e, sys)


def _prompt_user_input() -> dict:
    """
    Simple CLI prompts for a single customer input.
    Adjust the fields if your dataset columns differ.
    (customerid is dropped, churn is target)
    """
    print("\nEnter customer details for churn prediction.\n(Press Enter to accept default if shown)\n")

    def ask(name: str, default: str | None = None) -> str:
        prompt = f"{name}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        val = input(prompt).strip()
        return val if val else (default if default is not None else "")

    # Categorical fields
    data = {
        "gender": ask("gender (Male/Female)", "Male"),
        "partner": ask("partner (Yes/No)", "No"),
        "dependents": ask("dependents (Yes/No)", "No"),
        "phoneservice": ask("phoneservice (Yes/No)", "Yes"),
        "multiplelines": ask("multiplelines (Yes/No/No phone service)", "No"),
        "internetservice": ask("internetservice (DSL/Fiber optic/No)", "Fiber optic"),
        "onlinesecurity": ask("onlinesecurity (Yes/No/No internet service)", "No"),
        "onlinebackup": ask("onlinebackup (Yes/No/No internet service)", "No"),
        "deviceprotection": ask("deviceprotection (Yes/No/No internet service)", "No"),
        "techsupport": ask("techsupport (Yes/No/No internet service)", "No"),
        "streamingtv": ask("streamingtv (Yes/No/No internet service)", "No"),
        "streamingmovies": ask("streamingmovies (Yes/No/No internet service)", "No"),
        "contract": ask("contract (Month-to-month/One year/Two year)", "Month-to-month"),
        "paperlessbilling": ask("paperlessbilling (Yes/No)", "Yes"),
        "paymentmethod": ask(
            "paymentmethod (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))",
            "Electronic check"
        ),
    }

    # Numeric fields
    data["seniorcitizen"] = int(ask("seniorcitizen (0/1)", "0"))
    data["tenure"] = int(ask("tenure (months)", "12"))
    data["monthlycharges"] = float(ask("monthlycharges", "70.0"))
    data["totalcharges"] = float(ask("totalcharges", "840.0"))

    return data


def main() -> None:
    try:
        logging.info("Starting model prediction execution")
        predictor = ModelPredictor()

        print("Choose prediction mode:")
        print("1) Single input (interactive)")
        print("2) Batch predict from CSV")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "2":
            in_path = input("Path to input CSV (features only): ").strip()
            out_path = input("Output CSV path (optional, press Enter to skip): ").strip() or None

            result = predictor.predict_csv(in_path, out_path)
            print("\nDone. Preview:")
            print(result.head())
        else:
            user_data = _prompt_user_input()
            result = predictor.predict_one(user_data)

            label = "Churn (Yes)" if result["prediction"] == 1 else "No Churn"
            print("\n=== Prediction Result ===")
            print("Predicted:", label)
            print("Churn probability:", round(result["probability_churn"], 4))
            print("=========================")
    except Exception as e:
        logging.error("Exception occured in model prediction main execution")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
