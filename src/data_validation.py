import sys
from pathlib import Path
import pandas as pd
import yaml

from src.exception import CustomException
from src.logger import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataValidation:
    def __init__(
        self,
        raw_data_path: str | Path = PROJECT_ROOT / "data" / "telco_customer_churn.csv",
        schema_path: str | Path = PROJECT_ROOT / "config" / "schema.yaml",
        validated_dir: str | Path = PROJECT_ROOT / "validated_data",
        validated_filename: str = "telco_customer_churn_validated.csv",
        allow_extra_columns: bool = False,
        drop_missing_rows: bool = True,
        drop_duplicates: bool = True
    ) -> None:
        self.raw_data_path = Path(raw_data_path)
        self.schema_path = Path(schema_path)
        self.validated_dir = Path(validated_dir)
        self.validated_filename = validated_filename

        self.allow_extra_columns = allow_extra_columns
        self.drop_missing_rows = drop_missing_rows
        self.drop_duplicates = drop_duplicates

        self.schema = self._load_schema()
        self.expected_columns = [c.strip().lower() for c in self.schema["columns"]]
        self.drop_columns = [c.strip().lower() for c in self.schema.get("drop_columns", [])]
        self.numeric_columns = [c.strip().lower() for c in self.schema.get("numeric_columns", [])]
        self.target_column = self.schema["target_column"].strip().lower()

    def _load_schema(self) -> dict:
        try:
            if not self.schema_path.exists():
                raise FileNotFoundError(f"schema.yaml not found: {self.schema_path}")
            with open(self.schema_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        try:
            actual = set(df.columns)
            expected = set(self.expected_columns)

            missing = sorted(list(expected - actual))
            extra = sorted(list(actual - expected))

            if missing or (extra and not self.allow_extra_columns):
                msg = ["Schema validation failed:"]
                if missing:
                    msg.append(f"- Missing columns: {missing}")
                if extra and not self.allow_extra_columns:
                    msg.append(f"- Extra/unexpected columns: {extra}")
                raise ValueError("\n".join(msg))
        except Exception as e:
            raise CustomException(e, sys)

    def validate_and_save(self) -> pd.DataFrame:
        logging.info("Entered the validate_and_save method")
        try:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Raw data not found: {self.raw_data_path}")

            # 1) Read CSV
            logging.info("Reading raw data")
            df = pd.read_csv(self.raw_data_path)

            # 2) Normalize columns
            df = self._normalize_columns(df)

            # 3) Schema validation
            logging.info("Validating schema")
            self._validate_schema(df)

            # Keep only expected columns (stable schema)
            df = df[[c for c in self.expected_columns if c in df.columns]]

            rows_start = len(df)

            # Stats before cleaning
            missing_total_before = int(df.isna().sum().sum())
            dup_customerid_before = int(df.duplicated(subset=["customerid"]).sum()) if "customerid" in df.columns else 0
            dup_fullrow_before = int(df.duplicated().sum())

            # 4) Convert numeric columns properly (blanks -> NA -> numeric)
            logging.info("Converting numeric columns")
            for col in self.numeric_columns:
                if col in df.columns:
                    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            missing_total_after_conv = int(df.isna().sum().sum())

            # 5) Remove duplicates BEFORE dropping ID (based on customerid)
            if self.drop_duplicates and "customerid" in df.columns:
                df = df.drop_duplicates(subset=["customerid"])

            # 6) Remove missing rows
            if self.drop_missing_rows:
                df = df.dropna()

            # 7) Drop ID column(s) as per schema (customerid)
            for c in self.drop_columns:
                if c in df.columns:
                    df = df.drop(columns=[c])

            # 8) Remove duplicates AFTER dropping ID (this removes your "22 duplicates")
            dup_after_drop_id_before = int(df.duplicated().sum())
            if self.drop_duplicates:
                df = df.drop_duplicates()
            dup_after_drop_id_after = int(df.duplicated().sum())

            rows_final = len(df)

            # 9) Save validated data
            self.validated_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.validated_dir / self.validated_filename
            logging.info(f"Saving validated data to {out_path}")
            df.to_csv(out_path, index=False)

            # 10) Summary (Log summary instead of print)
            logging.info("=== Validation Summary ===")
            logging.info(f"Raw file: {self.raw_data_path}")
            logging.info(f"Validated file: {out_path}")
            logging.info(f"Rows (start): {rows_start}")

            logging.info("Missing values:")
            logging.info(f"Total missing (before conversion): {missing_total_before}")
            logging.info(f"Total missing (after conversion): {missing_total_after_conv}")

            logging.info("Duplicates:")
            logging.info(f"Duplicate customerid (before): {dup_customerid_before}")
            logging.info(f"Duplicate full rows (with customerid) before: {dup_fullrow_before}")
            logging.info(f"Duplicate rows after dropping customerid (before removal): {dup_after_drop_id_before}")
            logging.info(f"Duplicate rows after dropping customerid (after removal): {dup_after_drop_id_after}")

            logging.info(f"Rows (final): {rows_final}")
            logging.info("==========================")

            return df
        
        except Exception as e:
            raise CustomException(e, sys)


def main() -> None:
    try:
        logging.info("Starting data validation execution")
        validator = DataValidation(
            allow_extra_columns=False,
            drop_missing_rows=True,
            drop_duplicates=True
        )
        validator.validate_and_save()
    except Exception as e:
        logging.error("Exception occured in data validation main execution")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
