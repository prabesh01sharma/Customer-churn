# Customer Churn Prediction System

## Overview
An end-to-end customer churn prediction system featuring a machine learning pipeline, FastAPI backend, and modern web interface. Uses PostgreSQL data, Scikit-learn preprocessing, and Logistic Regression for classification with automatic model retraining capabilities.

## ✨ Features

### 🤖 ML Pipeline
- **Data Ingestion**: PostgreSQL integration for data fetching
- **Data Validation**: Schema validation, missing value handling, duplicate removal
- **Data Transformation**: Feature engineering, one-hot encoding, standard scaling
- **Model Training**: Logistic Regression with balanced class weights for churn prediction

### 🌐 Web Application
- **Modern UI**: Minimalistic dashboard with gradient design and card-based layout
- **Real-time Predictions**: Instant churn probability predictions via REST API
- **Model Retraining**: One-click model retraining with automatic performance comparison
- **Responsive Design**: Mobile-friendly interface with smooth animations

### 🔧 API Endpoints
- `GET /`: Serve frontend interface
- `POST /predict`: Predict customer churn probability
- `POST /retrain`: Trigger full pipeline and retrain model

## Project Structure
```
Customer-churn/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   └── retrain.py         # Model retraining orchestrator
├── frontend/              # Web interface
│   └── index.html         # Single-page application
├── src/                   # ML pipeline
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_train.py
│   └── model_prediction.py
├── config/                # Configuration files
│   ├── schema.yaml        # Data schema definition
│   └── config.py          # Database credentials
├── data/                  # Raw data
├── validated_data/        # Cleaned datasets
├── transformed_data/      # Preprocessed data + preprocessor
├── models/                # Trained model artifacts
├── logs/                  # Application logs
└── notebooks/             # Jupyter notebooks
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL database with customer data
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prabesh01sharma/Customer-churn.git
   cd Customer-churn
   ```

2. **Install dependencies**
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

3. **Configure database**
   Update credentials in `config/config.py` or set environment variables:
   ```bash
   export DB_USER=your_user
   export DB_PASSWORD=your_password
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=your_database
   ```

4. **Run the ML pipeline** (first time setup)
   ```bash
   python src/data_ingestion.py
   python src/data_validation.py
   python src/data_transformation.py
   python src/model_train.py
   ```

5. **Start the web application**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Open your browser**
   Navigate to `http://localhost:8000`

## 📊 Pipeline Stages

### 1. Data Ingestion (`src/data_ingestion.py`)
- Connects to PostgreSQL database
- Fetches telco customer churn data
- Saves raw data to `data/telco_customer_churn.csv`
- Logs execution details

### 2. Data Validation (`src/data_validation.py`)
- Validates against schema defined in `config/schema.yaml`
- Normalizes column names
- Converts numeric columns and handles missing values
- Removes duplicates based on `customerid`
- Saves validated data to `validated_data/`

### 3. Data Transformation (`src/data_transformation.py`)
- Separates features and target (`churn`)
- Performs stratified train-test split (80/20)
- Creates preprocessing pipeline:
  - One-hot encoding for categorical variables
  - Standard scaling for numerical variables
- Saves transformed data and fitted preprocessor to `transformed_data/`

### 4. Model Training (`src/model_train.py`)
- Trains Logistic Regression with `class_weight='balanced'`
- Evaluates on test set (accuracy, recall, precision, F1)
- Logs detailed classification report
- Saves model to `models/logistic_regression.pkl`

## 🎯 Using the Web Interface

### Making Predictions

1. Fill in customer information in the form
2. Click **"Predict Churn"**
3. View results:
   - **Churn prediction**: Yes/No
   - **Probability**: Visual bar + percentage
   - **Raw JSON response**

### Retraining the Model

1. Click **"Train Model"** button
2. Wait for pipeline to complete (~10-30 seconds)
3. Review metrics comparison:
   - Old vs New model performance
   - Decision: Model replaced or kept
   - Reason for decision

**Model Replacement Logic**:
- Replaces if recall improves by ≥2%
- Replaces if recall similar (±2%) AND accuracy improves
- Otherwise keeps existing model

## 🔌 API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "Female",
      "seniorcitizen": 0,
      "partner": "Yes",
      "dependents": "No",
      "tenure": 12,
      "phoneservice": "Yes",
      "multiplelines": "No",
      "internetservice": "Fiber optic",
      "onlinesecurity": "No",
      "onlinebackup": "No",
      "deviceprotection": "No",
      "techsupport": "No",
      "streamingtv": "Yes",
      "streamingmovies": "No",
      "contract": "Month-to-month",
      "paperlessbilling": "Yes",
      "paymentmethod": "Electronic check",
      "monthlycharges": 70.35,
      "totalcharges": 844.20
    }
  }'
```

**Response**:
```json
{
  "prediction": 1,
  "prediction_label": "Yes",
  "probability_churn": 0.8618
}
```

### Retrain Endpoint

```bash
curl -X POST http://localhost:8000/retrain
```

## 🎨 UI Features

- **Gradient Background**: Modern purple/indigo gradient
- **Card-based Layout**: Clean, organized sections
- **Icons**: Emoji icons for visual distinction
- **Animations**: Smooth hover effects and transitions
- **Responsive**: Mobile-friendly design
- **Probability Bar**: Visual gradient bar showing churn risk

## 📝 Model Performance

Current model (with balanced class weights):
- **Accuracy**: ~75%
- **Recall (Churn)**: ~78%
- **Precision**: Balanced for business use case

The model prioritizes **recall** to catch more potential churners, which is valuable for proactive customer retention.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn
- **ML**: Scikit-learn, Pandas, SQLAlchemy
- **Database**: PostgreSQL
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **Package Manager**: UV (or pip)

## 📄 Logging and Error Handling

- **Logging**: Implemented in `src/logger.py` with logs stored in `logs/app.log`
- **Exception Handling**: Custom exception class in `src/exception.py` captures detailed error information
- All pipeline stages utilize centralized logging and error handling

## 🎯 Conclusion

This project demonstrates a complete machine learning solution for customer churn prediction, from data ingestion and preprocessing to model deployment and real-time predictions. The system combines robust backend infrastructure with an intuitive user interface, making it easy to predict customer churn and continuously improve the model with new data.

Key achievements:
- **End-to-end ML pipeline** with automated data processing and model training
- **Production-ready API** for real-time predictions
- **Modern web interface** for easy user interaction
- **Automatic model retraining** with intelligent model comparison and replacement
- **Comprehensive logging** for debugging and monitoring

The balanced class weighting approach ensures high recall for churn cases, making this system practical for businesses focused on customer retention strategies.

