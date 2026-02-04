# Customer Churn Prediction

This project implements a machine learning pipeline to predict customer churn.

## Machine Learning Pipeline

### 1. Data Ingestion
- **Source**: PostgreSQL Database.
- **Implementation**: `src/data_ingestion.py` contains the `DataIngestion` class.
    - **Class**: `DataIngestion` handles connection to the database and saving data.
    - **Method**: `load_csv_data()` executes the query and saves the file.
- **Output**: Saves the raw dataset to `data/telco_customer_churn.csv`.

**Usage:**
Run the ingestion script directly:
```bash
python src/data_ingestion.py
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