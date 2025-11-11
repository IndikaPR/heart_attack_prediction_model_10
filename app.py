# FINAL_OPTIMIZED_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")
st.title("❤️ Heart Attack Risk Predictor - OPTIMIZED")

@st.cache_resource
def create_optimized_model():
    """Create a model that properly weights lifestyle risk factors"""
    
    # Enhanced training data with emphasis on lifestyle risks
    data = {
        'Age': [30, 24, 24, 27, 21, 20, 29, 32, 19, 35, 45, 28, 35, 55, 40, 25, 33],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'],
        'Region': ['East', 'East', 'North', 'East', 'West', 'West', 'East', 'North', 'West', 'South', 'North', 'East', 'North', 'North', 'West', 'North', 'North'],
        'Urban/Rural': ['Urban', 'Urban', 'Urban', 'Urban', 'Rural', 'Rural', 'Rural', 'Urban', 'Rural', 'Urban', 'Urban', 'Rural', 'Urban', 'Urban', 'Urban', 'Urban', 'Urban'],
        'SES': ['Middle', 'Low', 'Low', 'Middle', 'Low', 'Middle', 'High', 'Low', 'Middle', 'Middle', 'High', 'Middle', 'Low', 'Middle', 'High', 'Low', 'Low'],
        'Smoking Status': ['Never', 'Occasionally', 'Occasionally', 'Occasionally', 'Occasionally', 'Never', 'Regularly', 'Never', 'Occasionally', 'Never', 'Regularly', 'Occasionally', 'Occasionally', 'Regularly', 'Never', 'Occasionally', 'Occasionally'],
        'Alcohol Consumption': ['Regularly', 'Occasionally', 'Occasionally', 'Never', 'Occasionally', 'Never', 'Never', 'Occasionally', 'Occasionally', 'Never', 'Regularly', 'Occasionally', 'Occasionally', 'Regularly', 'Occasionally', 'Occasionally', 'Occasionally'],
        'Diet Type': ['Non-Vegetarian', 'Non-Vegetarian', 'Vegan', 'Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Vegetarian', 'Non-Vegetarian', 'Vegetarian', 'Vegetarian', 'Vegetarian'],
        'Physical Activity Level': ['Sedentary', 'Sedentary', 'High', 'Sedentary', 'Moderate', 'High', 'Moderate', 'Sedentary', 'Sedentary', 'Moderate', 'Sedentary', 'High', 'Sedentary', 'Sedentary', 'High', 'Sedentary', 'Sedentary'],
        'Screen Time (hrs/day)': [3, 15, 15, 6, 4, 2, 8, 13, 3, 5, 10, 4, 15, 12, 2, 14, 16],
        'Sleep Duration (hrs/day)': [8, 9, 3, 7, 9, 5, 10, 4, 9, 7, 5, 8, 3, 4, 8, 3, 4],
        'Family History of Heart Disease': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'Diabetes': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'],
        'Hypertension': ['Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'],
        'Cholesterol Levels (mg/dL)': [148, 124, 256, 137, 262, 205, 278, 254, 132, 180, 280, 160, 210, 280, 150, 220, 230],
        'BMI (kg/m²)': [34.4, 25.0, 33.9, 19.0, 28.0, 15.5, 21.4, 17.9, 18.2, 26.5, 35.0, 22.0, 28.0, 35.0, 22.5, 29.0, 30.0],
        'Stress Level': ['High', 'High', 'Low', 'Medium', 'Low', 'High', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low', 'Low', 'High', 'Low', 'Low', 'Medium'],
        'Blood Pressure (systolic/diastolic mmHg)': ['177/63', '137/110', '138/76', '177/90', '130/108', '171/107', '176/110', '146/76', '132/98', '130/85', '160/100', '120/80', '135/85', '160/100', '120/80', '140/90', '145/95'],
        'Resting Heart Rate (bpm)': [82, 76, 86, 106, 73, 115, 118, 71, 70, 72, 90, 65, 72, 95, 65, 85, 88],
        'ECG Results': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Normal'],
        'Chest Pain Type': ['Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal', 'Atypical', 'Atypical', 'Non-anginal', 'Atypical', 'Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal', 'Non-anginal', 'Typical', 'Non-anginal', 'Non-anginal', 'Atypical'],
        'Maximum Heart Rate Achieved': [183, 118, 164, 188, 216, 142, 181, 210, 161, 175, 140, 185, 180, 140, 185, 170, 165],
        'Exercise Induced Angina': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'],
        'Blood Oxygen Levels (SpO2%)': [94.1, 97.1, 92.7, 98.4, 94.9, 93.0, 93.4, 95.0, 90.9, 97.5, 92.0, 98.0, 97.0, 92.0, 98.5, 96.0, 95.0],
        'Triglyceride Levels (mg/dL)': [58, 341, 373, 102, 235, 129, 444, 316, 241, 150, 400, 120, 150, 400, 100, 200, 250],
        'Heart Attack Likelihood': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']  # Added more YES cases for lifestyle risks
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
    
    # Build optimized model with better architecture
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Use balanced class weights since we have fewer "Yes" cases
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Train longer with validation
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=16,
        verbose=0,
        validation_split=0.2,
        class_weight=class_weight_dict
    )
    
    return model, scaler, label_encoders, target_encoder, all_feature_columns

def main():
    st.write("### Creating Optimized Model...")
    
    # Create the optimized model
    model, scaler, label_encoders, target_encoder, feature_columns = create_optimized_model()
    
    st.success("✅ Optimized model created successfully!")
    
    # Input form
    st.write("### 📋 Enter Patient Information")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal & Lifestyle")
            age = st.number_input("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            screen_time = st.slider("Screen Time (hours/day)", 0, 24, 15)
            sleep = st.slider("Sleep Duration (hours/day)", 0, 24, 3)
            activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
            diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
            
        with col2:
            st.subheader("Medical History & Measurements")
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 210)
            bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 28.0)
            systolic = st.slider("Systolic BP", 80, 200, 135)
            diastolic = st.slider("Diastolic BP", 50, 150, 85)
            heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 72)
        
        # Other fields with defaults matching your patient
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
        triglycerides = 150.0
        
        submitted = st.form_submit_button("🔍 Predict Heart Attack Risk")
    
    if submitted:
        patient_data = {
            'Age': float(age), 'Gender': gender, 'Region': region, 'Urban/Rural': urban_rural, 'SES': ses,
            'Smoking Status': smoking, 'Alcohol Consumption': alcohol, 'Diet Type': diet,
            'Physical Activity Level': activity, 'Screen Time (hrs/day)': float(screen_time),
            'Sleep Duration (hrs/day)': float(sleep), 'Family History of Heart Disease': family_history,
            'Diabetes': diabetes, 'Hypertension': hypertension, 'Cholesterol Levels (mg/dL)': float(cholesterol),
            'BMI (kg/m²)': float(bmi), 'Stress Level': stress, 'Systolic_BP': float(systolic),
            'Diastolic_BP': float(diastolic), 'Resting Heart Rate (bpm)': float(heart_rate),
            'ECG Results': ecg, 'Chest Pain Type': chest_pain, 'Maximum Heart Rate Achieved': float(max_heart_rate),
            'Exercise Induced Angina': exercise_angina, 'Blood Oxygen Levels (SpO2%)': float(blood_oxygen),
            'Triglyceride Levels (mg/dL)': float(triglycerides)
        }
        
        # Process and predict
        feature_values = []
        for col in feature_columns:
            if col in ['Systolic_BP', 'Diastolic_BP']:
                feature_values.append(float(patient_data[col]))
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
                        feature_values.append(float(le.transform([value])[0]))
                    else:
                        feature_values.append(0.0)
            else:
                feature_values.append(0.0)
        
        # Convert to numpy array and predict
        feature_array = np.array([feature_values], dtype=np.float32)
        scaled_data = scaler.transform(feature_array)
        probability = model.predict(scaled_data, verbose=0)[0][0]
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        # Display results
        st.success(f"**Prediction:** {predicted_label}")
        st.metric("Heart Attack Probability", f"{probability:.1%}")
        
        # Enhanced risk analysis
        st.subheader("🔍 Comprehensive Risk Analysis")
        
        risk_score = 0
        risk_factors = []
        
        # Calculate risk score based on clinical guidelines
        if family_history == "Yes": 
            risk_score += 25
            risk_factors.append("🚨 Family history of heart disease (+25%)")
        if screen_time > 10: 
            risk_score += 20
            risk_factors.append(f"⚠️ High screen time: {screen_time} hrs/day (+20%)")
        if sleep < 6: 
            risk_score += 25
            risk_factors.append(f"🚨 Low sleep duration: {sleep} hrs/night (+25%)")
        if activity == "Sedentary": 
            risk_score += 15
            risk_factors.append("⚠️ Sedentary lifestyle (+15%)")
        if cholesterol > 200: 
            risk_score += 10
            risk_factors.append(f"⚠️ Borderline high cholesterol: {cholesterol} mg/dL (+10%)")
        if bmi > 25: 
            risk_score += 10
            risk_factors.append(f"⚠️ Overweight: BMI {bmi} (+10%)")
        if age > 40:
            risk_score += 5
            risk_factors.append(f"ℹ️ Age: {age} years (+5%)")
        
        st.write(f"**Calculated Risk Score:** {risk_score}%")
        
        for factor in risk_factors:
            st.write(f"• {factor}")
        
        # Final recommendation based on both model and clinical analysis
        st.subheader("💡 Medical Recommendation")
        
        if probability > 0.6 or risk_score > 50:
            st.error("""
            🚨 **MEDIUM-HIGH RISK** - Immediate Action Recommended:
            - Consult a cardiologist for comprehensive evaluation
            - Implement urgent lifestyle modifications
            - Increase sleep to 7-8 hours daily
            - Reduce screen time and increase physical activity
            - Monitor blood pressure and cholesterol regularly
            """)
        elif probability > 0.3 or risk_score > 30:
            st.warning("""
            ⚠️ **MODERATE RISK** - Preventive Measures Needed:
            - Schedule cardiac screening
            - Adopt heart-healthy lifestyle changes
            - Regular follow-up with healthcare provider
            - Focus on sleep improvement and activity increase
            """)
        else:
            st.success("""
            ✅ **LOW RISK** - Maintenance Recommended:
            - Continue healthy lifestyle habits
            - Regular annual check-ups
            - Maintain balanced diet and exercise
            """)

if __name__ == "__main__":
    main()
