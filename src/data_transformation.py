from __future__ import annotations

from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        if not self.validated_path.exists():
            raise FileNotFoundError(f"Validated data not found: {self.validated_path}")

        df = pd.read_csv(self.validated_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        # 1) Separate features and target
        y = (df[self.target_col] == "Yes").astype(int)  # Yes/No -> 1/0
        X = df.drop(columns=[self.target_col])

        # 2) Define numeric and categorical columns
        num_cols = [c for c in ["seniorcitizen", "tenure", "monthlycharges", "totalcharges"] if c in X.columns]
        cat_cols = [c for c in X.columns if c not in num_cols]

        # 3) Split first (avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # 4) Preprocess:
        # - OneHotEncode categorical
        # - StandardScale numeric
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", StandardScaler(), num_cols),
            ]
        )

        # Fit on train, transform train/test
        X_train_t = preprocess.fit_transform(X_train)
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

        X_train_df.to_csv(X_train_path, index=False)
        X_test_df.to_csv(X_test_path, index=False)
        y_train_df.to_csv(y_train_path, index=False)
        y_test_df.to_csv(y_test_path, index=False)

        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocess, f)

        print("=== Data Transformation Completed ===")
        print("Saved:")
        print(f"- {X_train_path}")
        print(f"- {X_test_path}")
        print(f"- {y_train_path}")
        print(f"- {y_test_path}")
        print(f"- {preprocessor_path}")
        print("====================================")

        return X_train_df, X_test_df, y_train_df, y_test_df


def main() -> None:
    transformer = DataTransformation()
    transformer.transform_and_save()


if __name__ == "__main__":
    main()
