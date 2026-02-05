import sys
from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataTransformation:
    def __init__(
        self,
        validated_path: str | Path = PROJECT_ROOT / "validated_data" / "telco_customer_churn_validated.csv",
        transformed_dir: str | Path = PROJECT_ROOT / "transformed_data",
        target_col: str = "churn",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.validated_path = Path(validated_path)
        self.transformed_dir = Path(transformed_dir)
        self.target_col = target_col.lower()
        self.test_size = test_size
        self.random_state = random_state

    def transform_and_save(self):
        logging.info("Entered the transform_and_save method")
        try:
            if not self.validated_path.exists():
                raise FileNotFoundError(f"Validated data not found: {self.validated_path}")

            logging.info(f"Reading validated data from {self.validated_path}")
            df = pd.read_csv(self.validated_path)
            df.columns = [c.strip().lower() for c in df.columns]

            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

            # 1) Separate features and target
            logging.info("Separating features and target")
            y = (df[self.target_col] == "Yes").astype(int)  # Yes/No -> 1/0
            X = df.drop(columns=[self.target_col])

            # 2) Define numeric and categorical columns
            num_cols = [c for c in ["seniorcitizen", "tenure", "monthlycharges", "totalcharges"] if c in X.columns]
            cat_cols = [c for c in X.columns if c not in num_cols]
            
            logging.info(f"Numeric columns: {num_cols}")
            logging.info(f"Categorical columns: {cat_cols}")

            # 3) Split first (avoid leakage)
            logging.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # 4) Preprocess:
            # - OneHotEncode categorical
            # - StandardScale numeric
            logging.info("Initializing ColumnTransformer for preprocessing")
            preprocess = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                    ("num", StandardScaler(), num_cols),
                ]
            )

            # Fit on train, transform train/test
            logging.info("Fitting and transforming training data")
            X_train_t = preprocess.fit_transform(X_train)
            logging.info("Transforming test data")
            X_test_t = preprocess.transform(X_test)

            # Feature names after transformation
            feature_names = preprocess.get_feature_names_out()

            # Convert to DataFrames (handle sparse matrices)
            X_train_df = pd.DataFrame(
                X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t,
                columns=feature_names
            )
            X_test_df = pd.DataFrame(
                X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t,
                columns=feature_names
            )

            y_train_df = pd.DataFrame(y_train, columns=[self.target_col])
            y_test_df = pd.DataFrame(y_test, columns=[self.target_col])

            # 5) Save outputs
            self.transformed_dir.mkdir(parents=True, exist_ok=True)

            X_train_path = self.transformed_dir / "X_train.csv"
            X_test_path = self.transformed_dir / "X_test.csv"
            y_train_path = self.transformed_dir / "y_train.csv"
            y_test_path = self.transformed_dir / "y_test.csv"
            preprocessor_path = self.transformed_dir / "preprocessor.pkl"
            
            logging.info(f"Saving transformed data to {self.transformed_dir}")
            X_train_df.to_csv(X_train_path, index=False)
            X_test_df.to_csv(X_test_path, index=False)
            y_train_df.to_csv(y_train_path, index=False)
            y_test_df.to_csv(y_test_path, index=False)

            with open(preprocessor_path, "wb") as f:
                pickle.dump(preprocess, f)

            logging.info("=== Data Transformation Completed ===")
            logging.info(f"Saved: {X_train_path}")
            logging.info(f"Saved: {X_test_path}")
            logging.info(f"Saved: {y_train_path}")
            logging.info(f"Saved: {y_test_path}")
            logging.info(f"Saved: {preprocessor_path}")
            logging.info("====================================")

            return X_train_df, X_test_df, y_train_df, y_test_df

        except Exception as e:
            raise CustomException(e, sys)


def main() -> None:
    try:
        logging.info("Starting data transformation execution")
        transformer = DataTransformation()
        transformer.transform_and_save()
    except Exception as e:
        logging.error("Exception occured in data transformation main execution")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
