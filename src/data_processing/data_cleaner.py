# src/data_processing/data_cleaner.py
from sklearn.preprocessing import LabelEncoder

class DataCleaner:
    def preprocess_data(self, df, target_column='prognosis'):
        """
        Prepare data for training
        Returns:
            X (DataFrame): Features
            y (Series): Encoded target
            le (LabelEncoder): Fitted label encoder
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return X, y_encoded, le