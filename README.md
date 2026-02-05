# Customer Churn Prediction

This project implements a machine learning pipeline to predict customer churn.

## Machine Learning Pipeline

### 1. Data Ingestion
- **Source**: PostgreSQL Database.
- **Implementation**: `src/data_ingestion.py` contains the `DataIngestion` class.
    - **Class**: `DataIngestion` handles connection to the database and saving data.
    - **Method**: `load_csv_data()` executes the query and saves the file.
- **Output**: Saves the raw dataset to `data/telco_customer_churn.csv`.

### 2. Data Validation
- **Input**: Raw CSV from `data/telco_customer_churn.csv`.
- **Implementation**: `src/data_validation.py` contains the `DataValidation` class.
    - **Class**: `DataValidation` handles schema validation, cleaning, and preprocessing.
    - **Method**: `validate_and_save()` executes validation logic (schema check, missing value handling, duplicate removal) and saves the file.
- **Output**: Saves validated dataset to `validated_data/telco_customer_churn_validated.csv`.

### 3. Data Transformation
- **Input**: Validated CSV from `validated_data/telco_customer_churn_validated.csv`.
- **Implementation**: `src/data_transformation.py` contains the `DataTransformation` class.
    - **Class**: `DataTransformation` handles splitting and feature engineering.
    - **Method**: `transform_and_save()` splits data (Train/Test) and applies `OneHotEncoder` (categorical) and `StandardScaler` (numeric).
- **Output**: Saves processed files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) and `preprocessor.pkl` to `transformed_data/`.

**Usage:**
Run the pipeline scripts in order:
```bash
python src/data_ingestion.py
python src/data_validation.py
python src/data_transformation.py
```

## Installation & Setup

### Prerequisites
- Python 3.x
- PostgreSQL database
- `uv` package manager

### Installation
1. Install `uv`:
   ```bash
   pip install uv
   ```
2. Initialize and install dependencies:
   ```bash
   uv sync
   ```

### Configuration
Create a `.env` file in the root directory with your database credentials:
```bash
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db
DB_SCHEMA=public
DB_TABLE=your_table
```