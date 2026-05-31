import os
import joblib
import pandas as pd
from src.utils import load_yaml, get_logger

logger = get_logger(__name__)

def batch_predict(input_csv_path: str, output_csv_path: str, config_path: str = "config/config.yaml"):
    """Performs batch prediction on a CSV file and saves the results."""
    logger.info("Starting batch prediction process...")
    cfg = load_yaml(config_path)
    
    df = pd.read_csv(input_csv_path)
    
    model = joblib.load(os.path.join(cfg['models']['dir'], "fraud_model.joblib"))
    scaler = joblib.load(os.path.join(cfg['models']['dir'], "scaler.joblib"))
    
    features = cfg['features']['numerical'] + cfg['features']['categorical']
    X = df[features]
    X_scaled = scaler.transform(X)
    
    df['predictions'] = model.predict(X_scaled)
    df['probability'] = model.predict_proba(X_scaled)[:, 1]
    
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Batch predictions saved successfully to {output_csv_path}")

if __name__ == "__main__":
    try:
        batch_predict("data/processed/test.csv", "data/processed/batch_predictions.csv")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")