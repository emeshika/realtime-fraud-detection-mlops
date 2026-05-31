import os
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import mlflow
import mlflow.sklearn

def load_config(config_path="config/config.yaml"):
    """Loads parameters and paths from the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_processed_data(processed_dir):
    """Loads the processed train, validation, and test datasets."""
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    print("Processed datasets loaded successfully for training.")
    return train_df, val_df, test_df

def separate_features_target(df, target_col):
    """Splits a dataframe into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_and_track_model():
    print("Model Training & Experiment Tracking Started...")
    
    # Load configuration
    cfg = load_config()
    target_col = cfg['features']['target']
    processed_dir = cfg['data']['processed_dir']
    
    # Load data
    train_df, val_df, test_df = load_processed_data(processed_dir)
    
    X_train, y_train = separate_features_target(train_df, target_col)
    X_val, y_val = separate_features_target(val_df, target_col)
    X_test, y_test = separate_features_target(test_df, target_col)
    
    # Set up MLflow Experiment name
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    # Start MLflow run to log hyperparameters and evaluation metrics
    with mlflow.start_run():
        print("MLflow Logging Session Initiated.")
        
        # Hyperparameters for Random Forest
        n_estimators = 100
        max_depth = 10
        random_state = cfg['data']['random_state']
        
        # Log Hyperparameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Initialize and Train Model
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight="balanced" # Handles data imbalance since fraud cases are typically low
        )
        model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Validate Model Performance
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"Validation Metrics -> Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f}")
        
        # Log Metrics to MLflow
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        mlflow.log_metric("val_f1_score", val_f1)
        
        # Evaluate on Test Data (Final Check)
        y_test_pred = model.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)
        mlflow.log_metric("test_f1_score", test_f1)
        
        # Log the trained model inside MLflow artifact store
        mlflow.sklearn.log_model(model, "fraud_model_artifact")
        print("Model logged as an MLflow artifact.")
        
        # Save the model locally inside models/ folder for API use
        os.makedirs("models", exist_ok=True)
        model_output_path = "models/fraud_model.joblib"
        joblib.dump(model, model_output_path)
        print(f"Trained model exported locally to: {model_output_path}")
        
        print("\n--- Detailed Classification Report (Test Data) ---")
        print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    train_and_track_model()
    print("Training pipeline run completed successfully!")