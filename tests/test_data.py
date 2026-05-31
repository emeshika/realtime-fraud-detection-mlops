import os
import pandas as pd
from src.utils import load_yaml

def test_processed_data_splits():
    """Validates that the train and test data splits exist and are valid."""
    cfg = load_yaml("config/config.yaml")
    processed_dir = cfg['data']['processed_dir']
    
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    assert os.path.exists(train_path), "train.csv does not exist!"
    assert os.path.exists(test_path), "test.csv does not exist!"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    assert not train_df.empty, "train.csv is empty!"
    assert not test_df.empty, "test.csv is empty!"
    
    target = cfg['features']['target']
    assert target in train_df.columns, f"Target '{target}' missing in train data"