# train_model.py
from src.data_ingestion.healthcare_data_loader import HealthcareDataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_analysis.disease_predictor import DiseasePredictor
import joblib

def main():
    # Load data
    loader = HealthcareDataLoader("data/Training.csv", "data/Testing.csv")
    train_df, test_df = loader.load_data()
    
    # Preprocess data
    cleaner = DataCleaner()
    X_train, y_train, le = cleaner.preprocess_data(train_df)
    X_test, y_test, _ = cleaner.preprocess_data(test_df)
    
    # Save artifacts
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")
    
    # Train model
    predictor = DiseasePredictor()
    predictor.train(X_train, y_train)
    
    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    print(f"Model Performance:\nAccuracy: {results['accuracy']:.2%}")
    print(f"F1-Score: {results['f1_score']:.2%}")
    
    # Save model
    predictor.save_model("models/disease_predictor.pkl")

if __name__ == "__main__":
    main()