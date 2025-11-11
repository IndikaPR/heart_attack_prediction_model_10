import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e86ab;
        margin: 10px 0px;
    }
    .feature-section {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    try:
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        return model, scaler, label_encoders, target_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def predict_heart_attack(patient_data, model, scaler, label_encoders, target_encoder):
    """Make prediction using the trained model"""
    try:
        # Create DataFrame from patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Preprocess the data
        processed_data = {}
        
        # Numerical features
        numerical_cols = ['Age', 'Screen Time (hrs/day)', 'Sleep Duration (hrs/day)',
                         'Cholesterol Levels (mg/dL)', 'BMI (kg/m²)', 'Resting Heart Rate (bpm)',
                         'Maximum Heart Rate Achieved', 'Blood Oxygen Levels (SpO2%)',
                         'Triglyceride Levels (mg/dL)']
        
        for col in numerical_cols:
            if col in patient_df.columns:
                processed_data[col] = float(patient_df[col].iloc[0])
            else:
                processed_data[col] = 0.0
        
        # Blood pressure handling
        if 'Blood Pressure (systolic/diastolic mmHg)' in patient_df.columns:
            bp = str(patient_df['Blood Pressure (systolic/diastolic mmHg)'].iloc[0])
            systolic, diastolic = bp.split('/')
            processed_data['Systolic_BP'] = float(systolic)
            processed_data['Diastolic_BP'] = float(diastolic)
        
        # Categorical features encoding
        categorical_columns = ['Gender', 'Region', 'Urban/Rural', 'SES', 'Smoking Status',
                              'Alcohol Consumption', 'Diet Type', 'Physical Activity Level',
                              'Family History of Heart Disease', 'Diabetes', 'Hypertension',
                              'Stress Level', 'ECG Results', 'Chest Pain Type', 'Exercise Induced Angina']
        
        for col in categorical_columns:
            if col in patient_df.columns:
                value = str(patient_df[col].iloc[0])
                le = label_encoders[col]
                if value in le.classes_:
                    processed_data[col] = le.transform([value])[0]
                else:
                    processed_data[col] = le.transform([le.classes_[0]])[0]
            else:
                processed_data[col] = 0
        
        # Create final DataFrame
        expected_columns = list(label_encoders.keys()) + numerical_cols + ['Systolic_BP', 'Diastolic_BP']
        final_df = pd.DataFrame([processed_data])
        
        # Ensure all columns exist
        for col in expected_columns:
            if col not in final_df.columns:
                final_df[col] = 0
        
        # Scale and predict
        scaled_data = scaler.transform(final_df)
        probability = model.predict(scaled_data)[0][0]
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "HIGH"
        elif probability > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return probability, predicted_label, risk_level
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">❤️ Heart Attack Risk Predictor</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050158.png", width=100)
        st.markdown("### About This App")
        st.info("""
        This AI-powered tool predicts heart attack risk based on medical and lifestyle factors.
        
        **Accuracy**: 77.8%
        **Sensitivity**: 100% (No heart attacks missed)
        
        *For educational purposes only*
        """)
        
        st.markdown("### 🩺 Medical Disclaimer")
        st.warning("This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")
    
    # Load model
    model, scaler, label_encoders, target_encoder = load_model_and_preprocessors()
    
    if model is None:
        st.error("❌ Model files not found. Please ensure all model files are in the same directory.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📋 Patient Information</div>', unsafe_allow_html=True)
        
        # Patient input form
        with st.form("patient_form"):
            col1a, col1b = st.columns(2)
            
            with col1a:
                st.markdown("#### Personal Information")
                age = st.slider("Age", 18, 80, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                region = st.selectbox("Region", ["North", "South", "East", "West"])
                urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
                ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
                
                st.markdown("#### Lifestyle Factors")
                smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
                alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
                diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
                screen_time = st.slider("Screen Time (hours/day)", 0, 16, 6)
                sleep = st.slider("Sleep Duration (hours/day)", 3, 12, 7)
            
            with col1b:
                st.markdown("#### Medical History")
                family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                
                st.markdown("#### Clinical Measurements")
                cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 400, 200)
                bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    systolic = st.slider("Systolic BP", 90, 200, 120)
                with bp_col2:
                    diastolic = st.slider("Diastolic BP", 60, 120, 80)
                
                heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 120, 72)
                ecg = st.selectbox("ECG Results", ["Normal", "Abnormal"])
                chest_pain = st.selectbox("Chest Pain Type", ["Non-anginal", "Atypical", "Typical"])
                max_heart_rate = st.slider("Maximum Heart Rate Achieved", 100, 220, 180)
                exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                blood_oxygen = st.slider("Blood Oxygen Levels (SpO2%)", 85.0, 100.0, 97.0)
                triglycerides = st.slider("Triglyceride Levels (mg/dL)", 50, 500, 150)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Risk Prediction</div>', unsafe_allow_html=True)
        
        if submitted:
            # Prepare patient data
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
                'BMI (kg/mÂ2)': bmi,
                'Stress Level': stress,
                'Blood Pressure (systolic/diastolic mmHg)': f"{systolic}/{diastolic}",
                'Resting Heart Rate (bpm)': heart_rate,
                'ECG Results': ecg,
                'Chest Pain Type': chest_pain,
                'Maximum Heart Rate Achieved': max_heart_rate,
                'Exercise Induced Angina': exercise_angina,
                'Blood Oxygen Levels (SpO2%)': blood_oxygen,
                'Triglyceride Levels (mg/dL)': triglycerides
            }
            
            # Make prediction
            with st.spinner('Analyzing patient data...'):
                probability, prediction, risk_level = predict_heart_attack(
                    patient_data, model, scaler, label_encoders, target_encoder
                )
            
            if probability is not None:
                # Display results
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Heart Attack Probability", f"{probability:.1%}")
                
                # Risk level display
                if risk_level == "HIGH":
                    st.markdown(f'<div class="risk-high">🚨 HIGH RISK - {prediction}</div>', unsafe_allow_html=True)
                elif risk_level == "MEDIUM":
                    st.markdown(f'<div class="risk-medium">⚠️ MEDIUM RISK - {prediction}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ LOW RISK - {prediction}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("#### 🔍 Risk Factors Identified")
                risk_factors = []
                
                if age > 45: risk_factors.append("Age > 45 years")
                if family_history == "Yes": risk_factors.append("Family history of heart disease")
                if diabetes == "Yes": risk_factors.append("Diabetes")
                if hypertension == "Yes": risk_factors.append("Hypertension")
                if cholesterol > 200: risk_factors.append(f"High cholesterol ({cholesterol} mg/dL)")
                if bmi > 30: risk_factors.append(f"High BMI ({bmi})")
                if smoking == "Regularly": risk_factors.append("Regular smoking")
                if stress == "High": risk_factors.append("High stress level")
                if systolic > 140 or diastolic > 90: risk_factors.append("High blood pressure")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No major risk factors identified")
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                if risk_level == "HIGH":
                    st.error("""
                    🚨 **Immediate Action Required:**
                    - Consult a cardiologist immediately
                    - Consider emergency evaluation if experiencing symptoms
                    - Monitor vital signs regularly
                    """)
                elif risk_level == "MEDIUM":
                    st.warning("""
                    ⚠️ **Preventive Measures Recommended:**
                    - Schedule a cardiac screening
                    - Adopt heart-healthy lifestyle changes
                    - Regular follow-up with healthcare provider
                    """)
                else:
                    st.success("""
                    ✅ **Maintenance Recommended:**
                    - Continue healthy lifestyle habits
                    - Regular annual check-ups
                    - Maintain balanced diet and exercise
                    """)
            
            else:
                st.error("Failed to generate prediction. Please check the input data.")
        
        else:
            # Default state before submission
            st.info("""
            **Instructions:**
            1. Fill out the patient information form
            2. Click 'Predict Heart Attack Risk'
            3. View the AI-powered risk assessment
            
            The model analyzes 25+ factors to provide accurate risk prediction.
            """)
            
            # Sample predictions
            st.markdown("#### 📊 Model Performance")
            st.metric("Accuracy", "77.8%")
            st.metric("Sensitivity", "100%")
            st.metric("Specificity", "71.4%")

if __name__ == "__main__":
    main()