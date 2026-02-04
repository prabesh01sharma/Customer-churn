from __future__ import annotations

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

from config.config import (
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_SCHEMA, DB_TABLE
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataIngestion:
    def __init__(
        self,
        output_dir: str | Path = PROJECT_ROOT / "data",
        output_filename: str = "telco_customer_churn.csv"
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_filename = output_filename

        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Basic validation
        if not self.output_filename.endswith(".csv"):
            raise ValueError("output_filename must end with .csv")

    def load_csv_data(self) -> pd.DataFrame:
        # Validate required env vars exist
        missing = [k for k, v in {
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
            "DB_NAME": DB_NAME,
            "DB_TABLE": DB_TABLE
        }.items() if not v]

        if missing:
            raise ValueError(f"Missing required env vars: {', '.join(missing)}")

        # Create DB engine
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        # Read PostgreSQL table into DataFrame
        query = f'SELECT * FROM {DB_SCHEMA}."{DB_TABLE}";'
        df = pd.read_sql(query, engine)

        # Save to /data folder
        out_path = self.output_dir / self.output_filename
        df.to_csv(out_path, index=False)

        print(f"Saved {len(df)} rows to: {out_path.resolve()}")
        return df

def main():
    ingestor = DataIngestion()
    ingestor.load_csv_data()


if __name__ == "__main__":
    main()