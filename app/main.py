# app/main.py
from __future__ import annotations

from pathlib import Path
import pickle
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import PredictRequest, PredictResponse
from app.retrain import ModelRetrainer
from src.logger import logging
from src.exception import CustomException

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "transformed_data" / "preprocessor.pkl"
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.yaml"
FRONTEND_PATH = PROJECT_ROOT / "frontend" / "index.html"

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


def load_yaml_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    schema = load_yaml_schema()
    return model, preprocessor, schema


# Load once when API starts
try:
    MODEL, PREPROCESSOR, SCHEMA = load_artifacts()
    logging.info("Model and preprocessor loaded successfully.")
except Exception as e:
    # App can still start but endpoints will error until files exist
    MODEL, PREPROCESSOR, SCHEMA = None, None, None
    logging.error(f"Failed to load artifacts: {e}")


@app.get("/", response_class=HTMLResponse)
def home():
    if not FRONTEND_PATH.exists():
        return "<h3>Frontend not found. Create frontend/index.html</h3>"
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.get("/required-features")
def required_features():
    """
    Returns the expected feature list, based on schema.yaml columns minus churn and drop_columns.
    """
    if SCHEMA is None:
        raise HTTPException(status_code=500, detail="Schema not loaded.")

    cols = [c.lower() for c in SCHEMA["columns"]]
    drop_cols = [c.lower() for c in SCHEMA.get("drop_columns", [])]
    target = SCHEMA["target_column"].lower()

    features = [c for c in cols if c not in drop_cols and c != target]
    return {"required_features": features}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Expects JSON: {"features": { ... }}
    Uses the same preprocessor.pkl + model.pkl to produce prediction.
    """
    try:
        if MODEL is None or PREPROCESSOR is None or SCHEMA is None:
            raise HTTPException(status_code=500, detail="Model/preprocessor/schema not loaded.")

        # Get required feature names from schema
        cols = [c.lower() for c in SCHEMA["columns"]]
        drop_cols = [c.lower() for c in SCHEMA.get("drop_columns", [])]
        target = SCHEMA["target_column"].lower()
        required = [c for c in cols if c not in drop_cols and c != target]

        # Normalize input keys to lowercase
        features = {k.strip().lower(): v for k, v in req.features.items()}

        missing = [c for c in required if c not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

        # Create one-row DataFrame in correct column order
        X = pd.DataFrame([[features[c] for c in required]], columns=required)

        # Transform using fitted preprocessor
        X_t = PREPROCESSOR.transform(X)

        proba = float(MODEL.predict_proba(X_t)[0, 1])
        pred = int(MODEL.predict(X_t)[0])
        label = "Yes" if pred == 1 else "No"

        return PredictResponse(
            prediction=pred,
            prediction_label=label,
            probability_churn=proba
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/retrain")
async def retrain_model():
    """
    Endpoint to trigger model retraining pipeline.
    
    Runs:
    1. Data ingestion
    2. Data validation
    3. Data transformation
    4. Model training
    5. Model comparison and replacement (if better)
    
    Returns comparison results and status.
    """
    try:
        logging.info("Received model retraining request")
        
        # Execute retraining
        retrainer = ModelRetrainer()
        result = retrainer.execute_retraining()
        
        # Reload model and preprocessor if replaced
        if result.get("model_replaced"):
            global MODEL, PREPROCESSOR
            MODEL, PREPROCESSOR = load_model_and_preprocessor()
            logging.info("Model and preprocessor reloaded successfully")
        
        return result
        
    except Exception as e:
        logging.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {e}")
