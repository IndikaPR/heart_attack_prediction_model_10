# app.py
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
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 10px 0px;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 10px 0px;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        margin: 10px 0px;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #2e86ab;
        margin: 15px 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0px;
    }
    .stButton button {
        width: 100%;
        background-color: #2e86ab;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    try:
        # Check if files exist
        required_files = [
            'heart_attack_prediction_final_model.h5',
            'scaler.pkl', 
            'label_encoders.pkl',
            'target_encoder.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"❌ Missing model files: {', '.join(missing_files)}")
            st.info("💡 Please run 'save_model_files.py' first to generate the model files.")
            return None, None, None, None
        
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        
        st.success("✅ Model loaded successfully!")
        return model, scaler, label_encoders, target_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
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
        
        # Reorder columns to match training
        final_df = final_df.reindex(columns=expected_columns, fill_value=0)
        
        # Scale and predict
        scaled_data = scaler.transform(final_df)
        probability = model.predict(scaled_data, verbose=0)[0][0]
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
        st.markdown("### 🏥 About This App")
        st.info("""
        This AI-powered tool predicts heart attack risk based on medical and lifestyle factors.
        
        **Model Performance:**
        - Accuracy: 77.8%
        - Sensitivity: 100%
        - Specificity: 71.4%
        """)
        
        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning("""
        This tool is for **educational purposes only**. 
        Always consult healthcare professionals for medical advice and diagnosis.
        """)
        
        st.markdown("### 📁 Required Files")
        if all(os.path.exists(f) for f in ['heart_attack_prediction_final_model.h5', 'scaler.pkl', 'label_encoders.pkl', 'target_encoder.pkl']):
            st.success("✅ All model files present")
        else:
            st.error("❌ Model files missing")
            st.info("Run 'save_model_files.py' first")
    
    # Load model
    model, scaler, label_encoders, target_encoder = load_model_and_preprocessors()
    
    if model is None:
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📋 Patient Information Form</div>', unsafe_allow_html=True)
        
        # Patient input form
        with st.form("patient_form"):
            col1a, col1b = st.columns(2)
            
            with col1a:
                st.markdown("#### 👤 Personal Information")
                age = st.slider("Age", 18, 80, 35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                region = st.selectbox("Region", ["North", "South", "East", "West"])
                urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
                ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
                
                st.markdown("#### 🏃 Lifestyle Factors")
                smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
                alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
                diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
                screen_time = st.slider("Screen Time (hours/day)", 0, 16, 6)
                sleep = st.slider("Sleep Duration (hours/day)", 3, 12, 7)
            
            with col1b:
                st.markdown("#### 🩺 Medical History")
                family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                
                st.markdown("#### 📊 Clinical Measurements")
                cholesterol = st.slider("Cholesterol Levels (mg/dL)", 100, 400, 200)
                bmi = st.slider("BMI (kg/m²)", 15.0, 40.0, 25.0)
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                
                st.markdown("**Blood Pressure**")
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
        st.markdown('<div class="sub-header">🎯 Risk Assessment</div>', unsafe_allow_html=True)
        
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
                'BMI (kg/m²)': bmi,
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
            with st.spinner('🔬 Analyzing patient data...'):
                probability, prediction, risk_level = predict_heart_attack(
                    patient_data, model, scaler, label_encoders, target_encoder
                )
            
            if probability is not None:
                # Display results
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                # Probability gauge
                st.metric("Heart Attack Probability", f"{probability:.1%}")
                
                # Progress bar for visual indication
                st.progress(float(probability))
                
                # Risk level display
                if risk_level == "HIGH":
                    st.markdown(f'<div class="risk-high">🚨 HIGH RISK</div>', unsafe_allow_html=True)
                    st.markdown(f"**Prediction:** {prediction}")
                elif risk_level == "MEDIUM":
                    st.markdown(f'<div class="risk-medium">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                    st.markdown(f"**Prediction:** {prediction}")
                else:
                    st.markdown(f'<div class="risk-low">✅ LOW RISK</div>', unsafe_allow_html=True)
                    st.markdown(f"**Prediction:** {prediction}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("#### 🔍 Identified Risk Factors")
                risk_factors = []
                
                if age > 45: risk_factors.append(f"Age ({age} years)")
                if family_history == "Yes": risk_factors.append("Family history of heart disease")
                if diabetes == "Yes": risk_factors.append("Diabetes")
                if hypertension == "Yes": risk_factors.append("Hypertension")
                if cholesterol > 200: risk_factors.append(f"High cholesterol ({cholesterol} mg/dL)")
                if bmi > 30: risk_factors.append(f"High BMI ({bmi})")
                if smoking == "Regularly": risk_factors.append("Regular smoking")
                if stress == "High": risk_factors.append("High stress level")
                if systolic > 140 or diastolic > 90: risk_factors.append("High blood pressure")
                if ecg == "Abnormal": risk_factors.append("Abnormal ECG")
                if exercise_angina == "Yes": risk_factors.append("Exercise induced angina")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("No major risk factors identified")
                
                # Recommendations
                st.markdown("#### 💡 Medical Recommendations")
                if risk_level == "HIGH":
                    st.error("""
                    🚨 **Immediate Action Required:**
                    - Consult a cardiologist immediately
                    - Consider emergency evaluation if experiencing symptoms
                    - Monitor vital signs regularly
                    - Avoid strenuous activities
                    """)
                elif risk_level == "MEDIUM":
                    st.warning("""
                    ⚠️ **Preventive Measures Recommended:**
                    - Schedule a comprehensive cardiac screening
                    - Adopt heart-healthy lifestyle changes
                    - Regular follow-up with healthcare provider
                    - Monitor blood pressure and cholesterol
                    """)
                else:
                    st.success("""
                    ✅ **Maintenance Recommended:**
                    - Continue healthy lifestyle habits
                    - Regular annual check-ups
                    - Maintain balanced diet and exercise routine
                    - Monitor risk factors periodically
                    """)
            
            else:
                st.error("Failed to generate prediction. Please check the input data.")
        
        else:
            # Default state before submission
            st.info("""
            **Instructions:**
            1. Fill out all patient information in the form
            2. Click **'Predict Heart Attack Risk'**
            3. View AI-powered risk assessment and recommendations
            
            The model analyzes **25+ medical and lifestyle factors** to provide accurate risk prediction.
            """)
            
            # Quick stats
            st.markdown("#### 📈 Model Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Accuracy", "77.8%")
            with col_stat2:
                st.metric("Sensitivity", "100%")
            with col_stat3:
                st.metric("Specificity", "71.4%")

if __name__ == "__main__":
    main()
