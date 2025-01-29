# app/pages/disease_predictor.py
import streamlit as st
import joblib
import pandas as pd

def show():
    st.title("Disease Prediction System")
    
    # Load artifacts
    model = joblib.load('models/disease_predictor.pkl')
    le = joblib.load('models/label_encoder.pkl')
    features = joblib.load('models/feature_names.pkl')
    
    # Symptom selection interface
    with st.form("prediction_form"):
        st.subheader("Select Symptoms")
        
        # Split symptoms into 3 columns for better layout
        col1, col2, col3 = st.columns(3)
        selected_symptoms = []
        
        with col1:
            for symptom in features[:44]:
                if st.checkbox(symptom.replace('_', ' ').title()):
                    selected_symptoms.append(symptom)
        
        with col2:
            for symptom in features[44:88]:
                if st.checkbox(symptom.replace('_', ' ').title()):
                    selected_symptoms.append(symptom)
        
        with col3:
            for symptom in features[88:]:
                if st.checkbox(symptom.replace('_', ' ').title()):
                    selected_symptoms.append(symptom)
        
        if st.form_submit_button("Predict Disease"):
            if len(selected_symptoms) < 3:
                st.warning("Please select at least 3 symptoms")
            else:
                # Create input vector
                input_data = pd.DataFrame(0, index=[0], columns=features)
                input_data[selected_symptoms] = 1
                
                # Make prediction
                prediction = model.predict(input_data)
                disease = le.inverse_transform(prediction)[0]
                
                # Show results
                st.success(f"**Predicted Disease:** {disease}")
                st.subheader("Selected Symptoms")
                st.write(", ".join([s.replace('_', ' ').title() for s in selected_symptoms]))