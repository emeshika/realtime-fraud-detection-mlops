import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI App
app = FastAPI(
    title="Real-Time Credit Card Fraud Detection Platform",
    description="Production-grade API to detect fraudulent credit card transactions instantly.",
    version="1.0.0"
)

# Define file paths for artifacts
MODEL_PATH = "models/fraud_model.joblib"
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

# Global variables to hold the loaded models
model = None
scaler = None
encoder = None

@app.on_event("startup")
def load_artifacts():
    """Loads ML model, scaler, and label encoder on API startup."""
    global model, scaler, encoder
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
        raise RuntimeWarning("Required ML artifacts (model, scaler, or encoder) are missing in 'models/' folder. Run pipeline first.")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("🚀 All ML artifacts loaded successfully. API is ready to predict!")

# Define the Input Schema using Pydantic for validation
class TransactionInput(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Fraud Detection API is running up and stable."}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """
    Accepts real-time transaction data, preprocesses it using saved components,
    and returns a fraud prediction.
    """
    try:
        # 1. Convert input JSON data to a Pandas DataFrame row
        input_data = pd.DataFrame([transaction.dict()])
        
        # 2. Categorical Encoding (Map merchant_category using trained Label Encoder)
        # Fallback mechanism if a totally new category arrives in real-time production
        try:
            input_data['merchant_category'] = encoder.transform(input_data['merchant_category'])
        except ValueError:
            # Assign a default unseen category encoding if brand new data shows up
            input_data['merchant_category'] = 0 
        
        # 3. Scale Numerical Features (Same mapping rules as Training phase)
        numerical_cols = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h", "cardholder_age"]
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # 4. Model Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] # Probability score of being fraud
        
        # 5. Formulate API JSON Response Output
        return {
            "is_fraud": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "verdict": "CRITICAL: Fraudulent Transaction Detected!" if prediction == 1 else "Safe Transaction Approved."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Failed: {str(e)}")