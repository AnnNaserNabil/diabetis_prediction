# src/data_ingestion/healthcare_data_loader.py
import pandas as pd

class HealthcareDataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
    
    def load_data(self):
        """Load training and testing datasets"""
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        return train_df, test_df