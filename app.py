# WORKING_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")
st.title("❤️ Heart Attack Risk Predictor - WORKING VERSION")

@st.cache_resource
def create_and_save_model():
    """Create a fresh model and save it"""
    # Enhanced training data with more realistic cases
    data = {
        'Age': [30, 24, 24, 27, 21, 20, 29, 32, 19, 35, 45, 28, 35, 55, 40],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female'],
        'Region': ['East', 'East', 'North', 'East', 'West', 'West', 'East', 'North', 'West', 'South', 'North', 'East', 'North', 'North', 'West'],
        'Urban/Rural': ['Urban', 'Urban', 'Urban', 'Urban', 'Rural', 'Rural', 'Rural', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Urban', 'Urban'],
        'SES': ['Middle', 'Low', 'Low', 'Middle', 'Low', 'Middle', 'High', 'Low', 'Middle', 'Middle', 'High', 'Middle', 'Low', 'Middle', 'High'],
        'Smoking Status': ['Never', 'Occasionally', 'Occasionally', 'Occasionally', 'Occasionally', 'Never', 'Regularly', 'Never', 'Occasionally', 'Never', 'Regularly', 'Occasionally', 'Occasionally', 'Regularly', 'Never'],
        'Alcohol Consumption': ['Regularly', 'Occasionally', 'Occasionally', 'Never', 'Occasionally', 'Never', 'Never', 'Occasionally', 'Occasionally', 'Never', 'Regularly', 'Occasionally', 'Occasionally', 'Regularly', 'Occasionally'],
        'Diet Type': ['Non-Vegetarian', 'Non-Vegetarian', 'Vegan', 'Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Vegetarian'],
        'Physical Activity Level': ['Sedentary', 'Sedentary', 'High', 'Sedentary', 'Moderate', 'High', 'Moderate', 'Sedentary', 'Sedentary', 'Moderate', 'Sedentary', 'High', 'Sedentary', 'Sedentary', 'High'],
        'Screen Time (hrs/day)': [3, 15, 15, 6, 4, 2, 8, 13, 3, 5, 10, 4, 15, 12, 2],
        'Sleep Duration (hrs/day)': [8, 9, 3, 7, 9, 5, 10, 4, 9, 7, 5, 8, 3, 4, 8],
        'Family History of Heart Disease': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
        'Diabetes': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'Hypertension': ['Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'Cholesterol Levels (mg/dL)': [148, 124, 256, 137, 262, 205, 278, 254, 132, 180, 280, 160, 210, 280, 150],
        'BMI (kg/m²)': [34.4, 25.0, 33.9, 19.0, 28.0, 15.5, 21.4, 17.9, 18.2, 26.5, 35.0, 22.0, 28.0, 35.0, 22.5],
        'Stress Level': ['High', 'High', 'Low', 'Medium', 'Low', 'High', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Low', 'High', 'Low'],
        'Blood Pressure (systolic/diastolic mmHg)': ['177/63', '137/110', '138/76', '177/90', '130/108', '171/107', '176/110', '146/76', '132/98', '130/85', '160/100', '120/80', '135/85', '160/100', '120/80'],
        'Resting Heart Rate (bpm)': [82, 76, 86, 106, 73, 115, 118, 71, 70, 72, 90, 65, 72, 95, 65],
        'ECG Results': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Abnormal', 'Normal'],
        'Chest Pain Type': ['Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal', 'Atypical', 'Atypical', 'Non-anginal', 'Atypical', 'Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal'],
        'Maximum Heart Rate Achieved': [183, 118, 164, 188, 216, 142, 181, 210, 161, 175, 140, 185, 180, 140, 185],
        'Exercise Induced Angina': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'Blood Oxygen Levels (SpO2%)': [94.1, 97.1, 92.7, 98.4, 94.9, 93.0, 93.4, 95.0, 90.9, 97.5, 92.0, 98.0, 97.0, 92.0, 98.5],
        'Triglyceride Levels (mg/dL)': [58, 341, 373, 102, 235, 129, 444, 316, 241, 150, 400, 120, 150, 400, 100],
        'Heart Attack Likelihood': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    
    # Preprocessing
    df_processed = df.copy()
    
    # Define categorical columns
    categorical_columns = ['Gender', 'Region', 'Urban/Rural', 'SES', 'Smoking Status',
                          'Alcohol Consumption', 'Diet Type', 'Physical Activity Level',
                          'Family History of Heart Disease', 'Diabetes', 'Hypertension',
                          'Stress Level', 'ECG Results', 'Chest Pain Type', 'Exercise Induced Angina']
    
    # Encode categorical variables
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column].astype(str))
        label_encoders[column] = le
    
    # Convert target variable
    target_encoder = LabelEncoder()
    df_processed['Heart Attack Likelihood'] = target_encoder.fit_transform(df_processed['Heart Attack Likelihood'])
    
    # Handle blood pressure
    df_processed['Systolic_BP'] = df_processed['Blood Pressure (systolic/diastolic mmHg)'].apply(lambda x: float(x.split('/')[0]))
    df_processed['Diastolic_BP'] = df_processed['Blood Pressure (systolic/diastolic mmHg)'].apply(lambda x: float(x.split('/')[1]))
    df_processed = df_processed.drop('Blood Pressure (systolic/diastolic mmHg)', axis=1)
    
    # Define numerical columns
    numerical_cols = ['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs/day)',
                     'Cholesterol Levels (mg/dL)', 'BMI (kg/m²)', 'Resting Heart Rate (bpm)',
                     'Maximum Heart Rate Achieved', 'Blood Oxygen Levels (SpO2%)',
                     'Triglyceride Levels (mg/dL)', 'Systolic_BP', 'Diastolic_BP']
    
    # Prepare features and target
    all_feature_columns = categorical_columns + numerical_cols
    X = df_processed[all_feature_columns]
    y = df_processed['Heart Attack Likelihood']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Build and train model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
    
    # Save the new model
    model.save('heart_attack_model_streamlit_fixed.h5')
    joblib.dump(scaler, 'scaler_streamlit_fixed.pkl')
    joblib.dump(label_encoders, 'label_encoders_streamlit_fixed.pkl')
    joblib.dump(target_encoder, 'target_encoder_streamlit_fixed.pkl')
    
    feature_order_info = {
        'all_feature_columns': all_feature_columns,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_cols
    }
    joblib.dump(feature_order_info, 'feature_order_streamlit_fixed.pkl')
    
    return model, scaler, label_encoders, target_encoder, feature_order_info

def predict_heart_attack_working(patient_data, model, scaler, label_encoders, target_encoder, feature_order_info):
    """Working prediction function"""
    try:
        all_feature_columns = feature_order_info['all_feature_columns']
        
        # Process each feature
        feature_values = []
        for col in all_feature_columns:
            if col in ['Systolic_BP', 'Diastolic_BP']:
                feature_values.append(float(patient_data[col]))
            elif col in patient_data:
                if col in feature_order_info['numerical_columns']:
                    feature_values.append(float(patient_data[col]))
                else:
                    value = str(patient_data[col])
                    le = label_encoders[col]
                    if value in le.classes_:
                        feature_values.append(float(le.transform([value])[0]))
                    else:
                        feature_values.append(0.0)
            else:
                feature_values.append(0.0)
        
        # Convert to numpy array
        feature_array = np.array([feature_values], dtype=np.float32)
        
        # Scale and predict
        scaled_data = scaler.transform(feature_array)
        probability = model.predict(scaled_data, verbose=0)[0][0]
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        return probability, predicted_label
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    st.write("### Creating fresh model in Streamlit environment...")
    
    # Create and load the new model
    model, scaler, label_encoders, target_encoder, feature_order_info = create_and_save_model()
    
    st.success("✅ Fresh model created and saved successfully!")
    
    # Test with the problematic patient
    st.write("### 🧪 Testing with your patient data...")
    
    test_patient = {
        'Age': 35, 'Gender': 'Male', 'Region': 'North', 'Urban/Rural': 'Urban', 'SES': 'Low',
        'Smoking Status': 'Occasionally', 'Alcohol Consumption': 'Occasionally', 'Diet Type': 'Vegetarian',
        'Physical Activity Level': 'Sedentary', 'Screen Time (hrs/day)': 15, 'Sleep Duration (hrs/day)': 3,
        'Family History of Heart Disease': 'Yes', 'Diabetes': 'No', 'Hypertension': 'No',
        'Cholesterol Levels (mg/dL)': 210, 'BMI (kg/m²)': 28.0, 'Stress Level': 'Low',
        'Systolic_BP': 135, 'Diastolic_BP': 85, 'Resting Heart Rate (bpm)': 72,
        'ECG Results': 'Normal', 'Chest Pain Type': 'Non-anginal', 'Maximum Heart Rate Achieved': 180,
        'Exercise Induced Angina': 'No', 'Blood Oxygen Levels (SpO2%)': 97.0, 'Triglyceride Levels (mg/dL)': 150
    }
    
    probability, prediction = predict_heart_attack_working(
        test_patient, model, scaler, label_encoders, target_encoder, feature_order_info
    )
    
    if probability is not None:
        st.success(f"**Prediction:** {prediction}")
        st.metric("Heart Attack Probability", f"{probability:.1%}")
        
        # Show risk factors
        st.subheader("🔍 Risk Factors Identified")
        risk_factors = []
        if test_patient['Family History of Heart Disease'] == 'Yes': 
            risk_factors.append("Family history of heart disease")
        if test_patient['Screen Time (hrs/day)'] > 10: 
            risk_factors.append(f"High screen time ({test_patient['Screen Time (hrs/day)']} hrs/day)")
        if test_patient['Sleep Duration (hrs/day)'] < 6: 
            risk_factors.append(f"Low sleep duration ({test_patient['Sleep Duration (hrs/day)']} hrs/night)")
        if test_patient['Physical Activity Level'] == 'Sedentary': 
            risk_factors.append("Sedentary lifestyle")
        if test_patient['Cholesterol Levels (mg/dL)'] > 200: 
            risk_factors.append(f"Borderline high cholesterol ({test_patient['Cholesterol Levels (mg/dL)']} mg/dL)")
        if test_patient['BMI (kg/m²)'] > 25: 
            risk_factors.append(f"Overweight (BMI: {test_patient['BMI (kg/m²)']})")
        
        for factor in risk_factors:
            st.write(f"• {factor}")
        
        if probability > 0.7:
            st.error("🚨 HIGH RISK - Consult a cardiologist immediately")
        elif probability > 0.4:
            st.warning("⚠️ MEDIUM RISK - Lifestyle changes and monitoring recommended")
        else:
            st.success("✅ LOW RISK - Maintain healthy habits with regular checkups")

if __name__ == "__main__":
    main()
