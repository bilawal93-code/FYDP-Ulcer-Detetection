from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Load your sklearn model
MODEL_PATH = "best_model_1.0.pkl"
model = joblib.load(MODEL_PATH)

# Define input schema
class SensorData(BaseModel):
    Temperature: float
    Pressure: float
    Heartrate: float

# FastAPI app
app = FastAPI(title="Bed Ulcer Predictor")

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/predict")
def predict(data: SensorData):
    try:
        # Prepare input
        X = pd.DataFrame([{
            'Temperature': float(data.Temperature),
            'Pressure': float(data.Pressure),
            'Heartrate': float(data.Heartrate)
        }])
        
        # Class labels
        ulcer_cat = ['Low', 'Medium', 'High']
        
        # Model prediction (numeric)
        pred_raw = int(model.predict(X)[0])   
        pred = ulcer_cat[pred_raw]            
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            probs_array = model.predict_proba(X)[0]
            prob_dict = {ulcer_cat[i]: float(probs_array[i]) for i in range(len(ulcer_cat))}
        else:
            prob_dict = None
        
        return {"prediction": pred, "probabilities": prob_dict}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
