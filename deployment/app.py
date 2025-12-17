import json
import joblib
import pandas as pd
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Define input model based on the user's JSON structure
class CharacterInput(BaseModel):
    region: str
    primary_role: str
    alignment: str
    status: str
    species: str
    honour_1to5: int
    ruthlessness_1to5: int
    intelligence_1to5: int
    combat_skill_1to5: int
    diplomacy_1to5: int
    leadership_1to5: int
    trait_strategic: Optional[bool] = False
    trait_impulsive: Optional[bool] = False
    trait_charismatic: Optional[bool] = False
    trait_vengeful: Optional[bool] = False
    trait_loyal: bool
    trait_scheming: bool
    feature_set_version: Optional[float] = 1.0

# Global variables to hold model and metadata
model = None
feature_columns = None

def load_artifacts():
    global model, feature_columns
    
    # Paths to artifacts (adjust these paths as needed)
    # We assume the user places the downloaded artifacts in a 'model_artifacts' folder
    model_path = os.path.join("model_artifacts", "model.joblib")
    features_path = os.path.join("model_artifacts", "feature_columns.json")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("WARNING: Model artifacts not found. Please place 'model.joblib' and 'feature_columns.json' in 'model_artifacts/'")
        return

    try:
        model = joblib.load(model_path)
        with open(features_path, "r") as f:
            feature_columns = json.load(f)["columns"]
        print("Model and metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(title="GOT House Predictor", lifespan=lifespan)

@app.post("/predict")
async def predict(character: CharacterInput):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    # Convert input to DataFrame
    data = character.dict()
    
    # Convert booleans to int (0/1) as expected by the model (if they were numeric in training)
    # Based on the CSV, traits were 0/1 integers.
    for key, value in data.items():
        if isinstance(value, bool):
            data[key] = 1 if value else 0
            
    df = pd.DataFrame([data])

    # Preprocessing: One-hot encoding
    # We must use the same dummy_na=True as in training
    df_encoded = pd.get_dummies(df, dummy_na=True)

    # Align with training columns
    # Reindex will add missing columns (filled with 0) and remove extra columns
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    try:
        prediction = model.predict(df_aligned)
        return {"house_affiliation": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
