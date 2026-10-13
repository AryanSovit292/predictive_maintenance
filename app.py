import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Jet Engine RUL Predictor", layout="wide", page_icon="✈️")

# --- Header ---
st.title("✈️ Jet Engine Predictive Maintenance")
st.markdown("Predict the **Remaining Useful Life (RUL)** of turbofan jet engines from time-series sensor data using an LSTM deep learning model.")

# --- Load Cached Artifacts ---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model_lstm.keras')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('feature_cols.joblib')
    return model, scaler, features

try:
    model, scaler, feature_cols = load_artifacts()
    st.sidebar.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- File Uploader ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Engine Sensor CSV", type=["csv", "txt"])

# Option to use sample engine from test set if no file uploaded
use_sample = st.sidebar.checkbox("Use sample test engine (Unit 1 from test set)", value=True if not uploaded_file else False)

df = None
if uploaded_file is not None:
    try:
        # Support both comma-separated and space-separated files
        df = pd.read_csv(uploaded_file, sep=r'\s+|,', engine='python')
    except Exception as e:
        st.error(f"Failed to read file: {e}")
elif use_sample:
    # Load default test sample
    index_cols = ['unit_number', 'time_cycles']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    cols = index_cols + setting_cols + sensor_cols
    full_test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=cols)
    df = full_test[full_test['unit_number'] == 1].copy()

# --- Prediction & Dashboard ---
if df is not None:
    st.subheader("Engine Operational Data")
    st.write(f"Total cycles in log: **{len(df)}**")
    st.dataframe(df.tail(5), use_container_width=True)

    seq_length = 30
    if len(df) < seq_length:
        st.warning(f"Engine log has fewer than {seq_length} cycles. Need at least {seq_length} cycles to predict.")
    else:
        # Preprocessing: Scale features and extract last 30 cycles
        recent_data = df[feature_cols].tail(seq_length).copy()
        scaled_data = scaler.transform(recent_data)
        
        # Reshape to (1, 30, num_features)
        input_seq = np.expand_dims(scaled_data, axis=0).astype(np.float32)
        
        # Predict RUL
        predicted_rul = float(model.predict(input_seq, verbose=0)[0][0])
        predicted_rul = max(0.0, predicted_rul)

        st.markdown("---")
        st.subheader("Maintenance Status & Prediction")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{predicted_rul:.1f} cycles")
            
            if predicted_rul > 50:
                st.success("🟢 **Status: Healthy**\nEngine is operating within normal parameters.")
            elif 20 < predicted_rul <= 50:
                st.warning("🟡 **Status: Warning**\nSchedule inspection within the next 20 cycles.")
            else:
                st.error("🔴 **Status: Critical**\nImmediate maintenance required before next flight!")

        with col2:
            st.markdown("**Recent Sensor Trajectories (Last 30 Cycles)**")
            fig, ax = plt.subplots(figsize=(8, 3))
            for s in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11']:
                if s in df.columns:
                    ax.plot(df['time_cycles'].tail(seq_length), df[s].tail(seq_length), label=s)
            ax.set_xlabel("Time Cycles")
            ax.set_ylabel("Sensor Value")
            ax.legend(loc="upper left")
            st.pyplot(fig)