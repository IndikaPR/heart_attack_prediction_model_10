# DEBUG_app.py - Let's find the exact issue
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os

st.set_page_config(page_title="Heart Attack Risk Predictor - DEBUG", layout="wide")
st.title("🔧 DEBUG MODE - Heart Attack Risk Predictor")

@st.cache_resource
def load_model_and_preprocessors():
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

def debug_prediction(patient_data, model, scaler, label_encoders, target_encoder, feature_order_info):
    try:
        all_feature_columns = feature_order_info['all_feature_columns']
        
        st.write("🔍 **DEBUG INFO:**")
        st.write(f"Total features expected: {len(all_feature_columns)}")
        st.write(f"First 10 features: {all_feature_columns[:10]}")
        
        # Process each feature
        feature_values = []
        st.write("📊 **PROCESSING EACH FEATURE:**")
        
        for i, col in enumerate(all_feature_columns):
            if col in ['Systolic_BP', 'Diastolic_BP']:
                value = float(patient_data[col])
                feature_values.append(value)
                st.write(f"{i:2d}. {col}: {value} (blood pressure)")
            elif col in patient_data:
                if col in ['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs/day)', 
                          'Cholesterol Levels (mg/dL)', 'BMI (kg/m²)', 'Resting Heart Rate (bpm)',
                          'Maximum Heart Rate Achieved', 'Blood Oxygen Levels (SpO2%)',
                          'Triglyceride Levels (mg/dL)']:
                    value = float(patient_data[col])
                    feature_values.append(value)
                    st.write(f"{i:2d}. {col}: {value} (numerical)")
                else:
                    value = str(patient_data[col])
                    le = label_encoders[col]
                    if value in le.classes_:
                        encoded = le.transform([value])[0]
                        feature_values.append(encoded)
                        st.write(f"{i:2d}. {col}: '{value}' → {encoded} (categorical)")
                    else:
                        feature_values.append(0)
                        st.write(f"{i:2d}. {col}: '{value}' → 0 (unknown category)")
            else:
                feature_values.append(0)
                st.write(f"{i:2d}. {col}: 0 (missing)")
        
        # Check the feature values
        st.write("🔢 **FEATURE VALUES SUMMARY:**")
        st.write(f"Number of features processed: {len(feature_values)}")
        st.write(f"Feature values: {feature_values}")
        
        # Convert to numpy array
        feature_array = np.array([feature_values])
        st.write(f"Array shape: {feature_array.shape}")
        
        # Check for zeros or unusual values
        zero_count = np.sum(feature_array == 0)
        st.write(f"Features with value 0: {zero_count}/{len(feature_values)}")
        
        # Scale the features
        st.write("⚖️ **SCALING FEATURES:**")
        scaled_data = scaler.transform(feature_array)
        st.write(f"Scaled data shape: {scaled_data.shape}")
        st.write(f"Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
        
        # Make prediction
        st.write("🤖 **MAKING PREDICTION:**")
        probability = model.predict(scaled_data, verbose=0)[0][0]
        st.write(f"Raw probability: {probability}")
        st.write(f"Probability percentage: {probability:.6%}")
        
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        return probability, predicted_label
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None, None

def main():
    model, scaler, label_encoders, target_encoder, feature_order_info = load_model_and_preprocessors()
    if model is None:
        return
    
    # Fixed patient data (same as your input)
    patient_data = {
        'Age': 35,
        'Gender': 'Male',
        'Region': 'North',
        'Urban/Rural': 'Urban', 
        'SES': 'Low',
        'Smoking Status': 'Occasionally',
        'Alcohol Consumption': 'Occasionally',
        'Diet Type': 'Vegetarian',
        'Physical Activity Level': 'Sedentary',
        'Screen Time (hrs/day)': 15,
        'Sleep Duration (hrs/day)': 3,
        'Family History of Heart Disease': 'Yes',
        'Diabetes': 'No',
        'Hypertension': 'No',
        'Cholesterol Levels (mg/dL)': 210,
        'BMI (kg/m²)': 28.0,
        'Stress Level': 'Low',
        'Systolic_BP': 135,
        'Diastolic_BP': 85,
        'Resting Heart Rate (bpm)': 72,
        'ECG Results': 'Normal',
        'Chest Pain Type': 'Non-anginal',
        'Maximum Heart Rate Achieved': 180,
        'Exercise Induced Angina': 'No',
        'Blood Oxygen Levels (SpO2%)': 97.0,
        'Triglyceride Levels (mg/dL)': 150
    }
    
    st.write("🎯 **TESTING FIXED PATIENT DATA:**")
    st.json(patient_data)
    
    probability, prediction = debug_prediction(patient_data, model, scaler, label_encoders, target_encoder, feature_order_info)
    
    if probability is not None:
        st.success("🎉 **PREDICTION COMPLETED**")
        st.metric("Heart Attack Probability", f"{probability:.1%}")
        st.metric("Prediction", prediction)
        
        if probability > 0.7:
            st.error("🚨 HIGH RISK")
        elif probability > 0.4:
            st.warning("⚠️ MEDIUM RISK") 
        else:
            st.success("✅ LOW RISK")

if __name__ == "__main__":
    main()
