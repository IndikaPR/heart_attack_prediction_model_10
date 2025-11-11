# FIXED app.py - SIMPLIFIED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    try:
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_order_info = joblib.load('feature_order.pkl')
        
        return model, scaler, label_encoders, target_encoder, feature_order_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

def predict_heart_attack_simple(patient_data, model, scaler, label_encoders, target_encoder, feature_order_info):
    """Simplified prediction function"""
    try:
        all_feature_columns = feature_order_info['all_feature_columns']
        
        # Process each feature in exact order
        feature_values = []
        for col in all_feature_columns:
            if col == 'Systolic_BP':
                feature_values.append(float(patient_data['Systolic_BP']))
            elif col == 'Diastolic_BP':
                feature_values.append(float(patient_data['Diastolic_BP']))
            elif col in patient_data:
                if col in ['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs/day)', 
                          'Cholesterol Levels (mg/dL)', 'BMI (kg/m²)', 'Resting Heart Rate (bpm)',
                          'Maximum Heart Rate Achieved', 'Blood Oxygen Levels (SpO2%)',
                          'Triglyceride Levels (mg/dL)']:
                    feature_values.append(float(patient_data[col]))
                else:
                    value = str(patient_data[col])
                    le = label_encoders[col]
                    if value in le.classes_:
                        feature_values.append(le.transform([value])[0])
                    else:
                        feature_values.append(0)
            else:
                feature_values.append(0)
        
        # Convert to numpy array and scale
        feature_array = np.array([feature_values])
        scaled_data = scaler.transform(feature_array)
        
        # Predict
        probability = model.predict(scaled_data, verbose=0)[0][0]
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        return probability, predicted_label
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    st.title("❤️ Heart Attack Risk Predictor")
    
    # Load model
    model, scaler, label_encoders, target_encoder, feature_order_info = load_model_and_preprocessors()
    if model is None:
        return
    
    # Input form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal & Lifestyle")
            age = st.number_input("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            screen_time = st.number_input("Screen Time (hours/day)", 0, 24, 15)
            sleep = st.number_input("Sleep Duration (hours/day)", 0, 24, 3)
            activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
            diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
            
        with col2:
            st.subheader("Medical History & Measurements")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 210)
            bmi = st.number_input("BMI (kg/m²)", 15.0, 40.0, 28.0)
            systolic = st.number_input("Systolic BP", 80, 200, 135)
            diastolic = st.number_input("Diastolic BP", 50, 150, 85)
            heart_rate = st.number_input("Resting Heart Rate (bpm)", 40, 120, 72)
        
        # Other required fields with defaults
        region = "North"
        urban_rural = "Urban"
        ses = "Low"
        smoking = "Occasionally"
        alcohol = "Occasionally"
        stress = "Low"
        ecg = "Normal"
        chest_pain = "Non-anginal"
        max_heart_rate = 180
        exercise_angina = "No"
        blood_oxygen = 97.0
        triglycerides = 150
        
        submitted = st.form_submit_button("Predict Risk")
    
    if submitted:
        patient_data = {
            'Age': age,
            'Gender': gender,
            'Region': region,
            'Urban/Rural': urban_rural,
            'SES': ses,
            'Smoking Status': smoking,
            'Alcohol Consumption': alcohol,
            'Diet Type': diet,
            'Physical Activity Level': activity,
            'Screen Time (hrs/day)': screen_time,
            'Sleep Duration (hrs/day)': sleep,
            'Family History of Heart Disease': family_history,
            'Diabetes': diabetes,
            'Hypertension': hypertension,
            'Cholesterol Levels (mg/dL)': cholesterol,
            'BMI (kg/m²)': bmi,
            'Stress Level': stress,
            'Systolic_BP': systolic,
            'Diastolic_BP': diastolic,
            'Resting Heart Rate (bpm)': heart_rate,
            'ECG Results': ecg,
            'Chest Pain Type': chest_pain,
            'Maximum Heart Rate Achieved': max_heart_rate,
            'Exercise Induced Angina': exercise_angina,
            'Blood Oxygen Levels (SpO2%)': blood_oxygen,
            'Triglyceride Levels (mg/dL)': triglycerides
        }
        
        probability, prediction = predict_heart_attack_simple(
            patient_data, model, scaler, label_encoders, target_encoder, feature_order_info
        )
        
        if probability is not None:
            st.success(f"**Prediction:** {prediction}")
            st.metric("Heart Attack Probability", f"{probability:.1%}")
            
            if probability > 0.7:
                st.error("🚨 HIGH RISK - Consult a doctor immediately")
            elif probability > 0.4:
                st.warning("⚠️ MEDIUM RISK - Lifestyle changes recommended")
            else:
                st.success("✅ LOW RISK - Maintain healthy habits")

if __name__ == "__main__":
    main()
