import os
import pandas as pd
import yaml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def load_config(config_path="config/config.yaml"):
    """Loads configuration file paths."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_drift_report():
    print("🔍 Data Drift Detection Process Started...")
    
    cfg = load_config()
    processed_dir = cfg['data']['processed_dir']
    
    # 1. Load Baseline Data (The training data used as a reference)
    train_path = os.path.join(processed_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Baseline train.csv not found. Please run the data pipeline first.")
    
    reference_df = pd.read_csv(train_path)
    
    # 2. Simulate Production/Incoming Data
    test_path = os.path.join(processed_dir, "test.csv")
    current_df = pd.read_csv(test_path)
    
    # Injecting artificial drift into the 'amount' feature for simulation
    current_df['amount'] = current_df['amount'] * 1.5 
    
    # Drop target column 'is_fraud' to simulate unseen production data
    target_col = cfg['features']['target']
    if target_col in reference_df.columns:
        reference_df = reference_df.drop(columns=[target_col])
    if target_col in current_df.columns:
        current_df = current_df.drop(columns=[target_col])

    print("Evaluating data distributions between Training and Production batches...")
    
    # 3. Initialize Evidently Data Drift Report
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Execute the comparison
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # 4. Save report as an interactive HTML dashboard
    os.makedirs("monitoring", exist_ok=True)
    report_output_path = "monitoring/data_drift_report.html"
    drift_report.save_html(report_output_path)
    
    print(f"Data Drift Report generated successfully and saved to: {report_output_path}")

if __name__ == "__main__":
    generate_drift_report()