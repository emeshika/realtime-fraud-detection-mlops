import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_config(config_path="config/config.yaml"):
    """
    Loads the YAML configuration file containing paths and hyperparameters.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_data(file_path):
    """
    Loads the raw credit card transaction dataset from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

def preprocess_data(df, config, save_artifacts=True):
    """
    Cleans raw data and encodes categorical text variables into numbers.
    Also saves the label encoder for real-time inference later.
    """
    # 1. Drop unnecessary columns that don't contribute to fraud patterns
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])
        
    # 2. Handle missing values (Crucial fail-safe for production pipelines)
    df = df.dropna()

    # 3. Encode Categorical Features (Transforming text to numerical values)
    # We save the encoder artifact to ensure real-time API data is mapped identical to training data.
    le = LabelEncoder()
    df['merchant_category'] = le.fit_transform(df['merchant_category'])
    
    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(le, "models/label_encoder.joblib")
        print("Label Encoder saved successfully to 'models/' directory.")

    return df

def split_and_scale_data(df, config):
    """
    Splits data into Train, Validation, and Test sets, then standardizes 
    numerical scales to prevent data leakage and improve model performance.
    """
    target_col = config['features']['target']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into Train and Test sets (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y  # Maintains equal fraud ratios across subsets
    )

    # Further split Train to create a Validation set (10% of remaining training data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=config['data']['val_size'], 
        random_state=config['data']['random_state'],
        stratify=y_train
    )

    # Feature Scaling: Normalize numerical features to a uniform distribution
    scaler = StandardScaler()
    num_cols = config['features']['numerical']
    
    # Fit ONLY on training data to strictly prevent data leakage
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save the scaler object for the real-time prediction API
    joblib.dump(scaler, "models/scaler.joblib")
    print("Standard Scaler saved successfully to 'models/' directory.")

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, output_dir):
    """
    Saves the final train, validation, and test datasets as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Recombine features and targets before saving
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Processed datasets saved successfully to: {output_dir}")
    print(f"Dataset Shapes -> Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    print("Execution Started: Data Processing Pipeline")
    
    # Initialize configuration parameters
    cfg = load_config()
    
    # Step 1: Ingest Raw Data
    raw_df = load_data(cfg['data']['raw_path'])
    
    # Step 2: Clean and Preprocess
    preprocessed_df = preprocess_data(raw_df, cfg)
    
    # Step 3: Split and Scale features
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_and_scale_data(preprocessed_df, cfg)
    
    # Step 4: Export final files
    save_processed_data(X_tr, X_va, X_te, y_tr, y_va, y_te, cfg['data']['processed_dir'])
    
    print("Execution Completed: Data Pipeline Finished Successfully!")