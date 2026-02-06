# app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any


class PredictRequest(BaseModel):
    # generic dict-based input so you don't have to hardcode 19 fields here
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int                  # 0/1
    prediction_label: str            # "No" / "Yes"
    probability_churn: float         # 0.0 - 1.0
