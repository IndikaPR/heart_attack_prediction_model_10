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
            'target_encoder.pkl',
            'feature_order.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"❌ Missing model files: {', '.join(missing_files)}")
            st.info("💡 Please run 'save_model_files.py' first to generate the model files.")
            return None, None, None, None, None
        
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_order_info = joblib.load('feature_order.pkl')
        
        st.success("✅ Model loaded successfully!")
        return model, scaler, label_encoders, target_encoder, feature_order_info
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None, None

def predict_heart_attack(patient_data, model, scaler, label_encoders, target_encoder, feature_order_info):
    """Make prediction using the trained model with exact column order"""
    try:
        # Get the exact feature order from training
        all_feature_columns = feature_order_info['all_feature_columns']
        categorical_columns = feature_order_info['categorical_columns']
        numerical_columns = feature_order_info['numerical_columns']
        
        # Create a dictionary to store processed data
        processed_data = {}
        
        # Step 1: Extract blood pressure values first
        if 'Blood Pressure (systolic/diastolic mmHg)' in patient_data:
            bp = str(patient_data['Blood Pressure (systolic/diastolic mmHg)'])
            systolic, diastolic = bp.split('/')
            processed_data['Systolic_BP'] = float(systolic)
            processed_data['Diastolic_BP'] = float(diastolic)
        
        # Step 2: Process ALL features in the EXACT same order as training
        for col in all_feature_columns:
            if col in patient_data:
                # This column was provided by user
                if col in numerical_columns:
                    # Numerical column - convert to float
                    processed_data[col] = float(patient_data[col])
                elif col in categorical_columns:
                    # Categorical column - encode it
                    value = str(patient_data[col])
                    le = label_encoders[col]
                    if value in le.classes_:
                        processed_data[col] = le.transform([value])[0]
                    else:
                        # Use first category as default for unknown values
                        processed_data[col] = le.transform([le.classes_[0]])[0]
            else:
                # Column not provided - use default value
                if col in numerical_columns:
                    processed_data[col] = 0.0
                else:
                    processed_data[col] = 0
        
        # Step 3: Create DataFrame with EXACT same column order and names as training
        final_df = pd.DataFrame([processed_data])
        
        # Ensure the DataFrame has the exact same columns in the exact same order
        final_df = final_df.reindex(columns=all_feature_columns, fill_value=0)
        
        # Debug: Show the final DataFrame structure
        st.write("🔍 Debug: Final DataFrame columns:", final_df.columns.tolist())
        st.write("🔍 Debug: Final DataFrame shape:", final_df.shape)
        
        # Step 4: Scale the features
        scaled_data = scaler.transform(final_df)
        
        # Step 5: Make prediction
        probability = model.predict(scaled_data, verbose=0)[0][0]
        prediction = (probability > 0.5).astype(int)
        predicted_label = target_encoder.inverse_transform([prediction])[0]
        
        # Step 6: Determine risk level
        if probability > 0.7:
            risk_level = "HIGH"
        elif probability > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return probability, predicted_label, risk_level
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
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
    
    # Load model
    model, scaler, label_encoders, target_encoder, feature_order_info = load_model_and_preprocessors()
    
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
                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                region = st.selectbox("Region", ["North", "South", "East", "West"])
                urban_rural = st.selectbox("Urban/Rural", ["Urban", "Rural"])
                ses = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
                
                st.markdown("#### 🏃 Lifestyle Factors")
                smoking = st.selectbox("Smoking Status", ["Never", "Occasionally", "Regularly"])
                alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
                diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                activity = st.selectbox("Physical Activity Level", ["Sedentary", "Moderate", "High"])
                screen_time = st.number_input("Screen Time (hours/day)", min_value=0, max_value=16, value=6)
                sleep = st.number_input("Sleep Duration (hours/day)", min_value=3, max_value=12, value=7)
            
            with col1b:
                st.markdown("#### 🩺 Medical History")
                family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                
                st.markdown("#### 📊 Clinical Measurements")
                cholesterol = st.number_input("Cholesterol Levels (mg/dL)", min_value=100, max_value=400, value=200)
                bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=40.0, value=25.0)
                stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
                
                st.markdown("**Blood Pressure**")
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    systolic = st.number_input("Systolic BP", min_value=90, max_value=200, value=120)
                with bp_col2:
                    diastolic = st.number_input("Diastolic BP", min_value=60, max_value=120, value=80)
                
                heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=50, max_value=120, value=72)
                ecg = st.selectbox("ECG Results", ["Normal", "Abnormal"])
                chest_pain = st.selectbox("Chest Pain Type", ["Non-anginal", "Atypical", "Typical"])
                max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=100, max_value=220, value=180)
                exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                blood_oxygen = st.number_input("Blood Oxygen Levels (SpO2%)", min_value=85.0, max_value=100.0, value=97.0)
                triglycerides = st.number_input("Triglyceride Levels (mg/dL)", min_value=50, max_value=500, value=150)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Heart Attack Risk", use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Risk Assessment</div>', unsafe_allow_html=True)
        
        if submitted:
            # Prepare patient data dictionary
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
                    patient_data, model, scaler, label_encoders, target_encoder, feature_order_info
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
                elif risk_level == "MEDIUM":
                    st.markdown(f'<div class="risk-medium">⚠️ MEDIUM RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ LOW RISK</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Prediction:** {prediction}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display success message
                st.success("✅ Prediction completed successfully!")
                
            else:
                st.error("❌ Failed to generate prediction. Please check the input data.")
        
        else:
            # Default state before submission
            st.info("""
            **Instructions:**
            1. Fill out all patient information in the form
            2. Click **'Predict Heart Attack Risk'**
            3. View AI-powered risk assessment and recommendations
            
            The model analyzes **25+ medical and lifestyle factors** to provide accurate risk prediction.
            """)

if __name__ == "__main__":
    main()
