# NUCLEAR_FIX_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

st.set_page_config(page_title="Heart Attack Risk - NUCLEAR FIX", layout="wide")
st.title("❤️ Heart Attack Risk Predictor - DIRECT TEST")

def nuclear_test():
    """Direct test bypassing all preprocessing"""
    
    # Load everything fresh
    model = keras.models.load_model('heart_attack_prediction_final_model.h5')
    scaler = joblib.load('scaler.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    
    # Test with EXACT same values that worked in Colab
    # From Colab: Probability: 0.619985 (61.9985%)
    test_features = np.array([[
        1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0,  # Categorical
        0.0, 1.0, 1.0, 2.0, 0.0, 35.0, 15.0, 3.0, 210.0,   # Mixed
        28.0, 72.0, 180.0, 97.0, 150.0, 135.0, 85.0         # Numerical
    ]], dtype=np.float32)
    
    st.write("🧪 **DIRECT MODEL TEST**")
    st.write(f"Input shape: {test_features.shape}")
    st.write(f"Input dtype: {test_features.dtype}")
    
    # Scale
    scaled_data = scaler.transform(test_features)
    st.write(f"Scaled data range: [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")
    
    # Predict
    probability = model.predict(scaled_data, verbose=0)[0][0]
    prediction = (probability > 0.5).astype(int)
    predicted_label = target_encoder.inverse_transform([prediction])[0]
    
    st.write(f"🎯 **RESULTS:**")
    st.write(f"Raw probability: {probability:.6f}")
    st.write(f"Percentage: {probability:.4%}")
    st.write(f"Prediction: {predicted_label}")
    
    return probability, predicted_label

def main():
    st.write("### Testing with exact same data that worked in Colab...")
    
    probability, prediction = nuclear_test()
    
    if probability > 0.6:
        st.success("✅ **MODEL IS WORKING CORRECTLY**")
        st.error("🚨 **ISSUE IS IN STREAMLIT DATA COLLECTION/PROCESSING**")
    else:
        st.error("❌ **MODEL IS BROKEN IN STREAMLIT ENVIRONMENT**")
    
    # Test different scenarios
    st.write("---")
    st.write("### 🔧 Additional Debugging")
    
    # Test 1: Simple prediction
    st.write("**Test 1: Simple input**")
    try:
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        test_simple = np.ones((1, 26), dtype=np.float32)
        result = model.predict(test_simple, verbose=0)[0][0]
        st.write(f"All ones prediction: {result:.6f}")
    except Exception as e:
        st.error(f"Simple test failed: {e}")
    
    # Test 2: Check model architecture
    st.write("**Test 2: Model Info**")
    try:
        model = keras.models.load_model('heart_attack_prediction_final_model.h5')
        st.write(f"Input shape: {model.input_shape}")
        st.write(f"Output shape: {model.output_shape}")
        st.write(f"Layers: {len(model.layers)}")
    except Exception as e:
        st.error(f"Model info failed: {e}")

if __name__ == "__main__":
    main()
