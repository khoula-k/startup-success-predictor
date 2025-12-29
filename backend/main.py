from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from fastapi.middleware.cors import CORSMiddleware

FEATURE_ORDER = [
    "state_code",
    #"zip_code",
    "city",
    # "first_funding_at",
    # "last_funding_at",
    "age_first_funding_year",
    "age_last_funding_year",
    "relationships",
    "funding_rounds",
    "funding_total_usd",
    "milestones",
    "category_code",
    "avg_participants",
    "is_top500",
    "has_RoundABCD",
    "has_Investor",
    "has_Seed"
]

# Path to model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "random_forest_model.joblib"


# Load model at startup
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Allow frontend access (React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace * with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema definition
class ModelInput(BaseModel):
    state_code: str
    city: str
    age_first_funding_year: float
    age_last_funding_year: float
    relationships: int
    funding_rounds: int
    funding_total_usd: float
    milestones: int
    category_code: str
    avg_participants: float
    is_top500: int
    has_RoundABCD: int
    has_Investor: int
    has_Seed: int

# Health check
@app.get("/")
def health_check():
    return {"status": "running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: ModelInput):
    # Convert the Pydantic model to a standard Python dictionary
    input_dict = data.model_dump()

    # Create the DataFrame using the dictionary keys
    ordered_values = [input_dict[f] for f in FEATURE_ORDER]

    # Create DataFrame with correct column names
    X = pd.DataFrame([ordered_values], columns=FEATURE_ORDER)
    print(f"Input data for prediction:\n{X}")

    # Predict
    prediction = model.predict(X)       
    probability = model.predict_proba(X)

    return {
        "prediction": prediction.tolist(),
        "probability": probability.tolist()
    }
