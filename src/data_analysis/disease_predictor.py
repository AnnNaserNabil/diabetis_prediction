# src/data_analysis/disease_predictor.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

class DiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=150, 
                                          class_weight='balanced',
                                          random_state=42)
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        preds = self.model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, preds),
            'f1_score': f1_score(y_test, preds, average='weighted')
        }
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load pretrained model"""
        self.model = joblib.load(path)