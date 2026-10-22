import os
import pickle
import struct
import zipfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Jet Engine Predictive Maintenance", layout="wide")


class NumpyLSTMModel:
    """Pure-NumPy inference fallback for model_lstm.keras.

    Used automatically on Windows 11 when Smart App Control blocks TensorFlow's
    C++ DLLs (_pywrap_debug_events_writer.pyd), producing identical predictions
    without requiring Smart App Control to be turned off.
    """

    def __init__(self, keras_path):
        with zipfile.ZipFile(keras_path, "r") as z:
            h5 = z.read("model.weights.h5")

        blocks = {}
        for i in range(len(h5) - 18):
            if h5[i] == 3 and h5[i + 1] == 1:
                addr, size = struct.unpack("<QQ", h5[i + 2 : i + 18])
                if size in (12000, 40000, 800, 200, 4) and 0 < addr < len(h5) - size:
                    blocks[size] = np.frombuffer(
                        h5[addr : addr + size], dtype=np.float32
                    ).copy()

        self.W_lstm = blocks[12000].reshape(15, 200)
        self.U_lstm = blocks[40000].reshape(50, 200)
        self.b_lstm = blocks[800].reshape(200)
        self.W_dense = blocks[200].reshape(50, 1)
        self.b_dense = blocks[4].reshape(1)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

    def predict(self, X, verbose=0):
        X = np.asarray(X, dtype=np.float32)
        batch_size, time_steps, _ = X.shape
        h = np.zeros((batch_size, 50), dtype=np.float32)
        c = np.zeros((batch_size, 50), dtype=np.float32)

        for t in range(time_steps):
            z = X[:, t, :] @ self.W_lstm + h @ self.U_lstm + self.b_lstm
            i = self._sigmoid(z[:, 0:50])
            f = self._sigmoid(z[:, 50:100])
            g = np.tanh(z[:, 100:150])
            o = self._sigmoid(z[:, 150:200])
            c = f * c + i * g
            h = o * np.tanh(c)

        return h @ self.W_dense + self.b_dense


class NumpyMinMaxScaler:
    """Fallback scaler if scikit-learn DLLs are ever blocked by Smart App Control."""

    def __init__(self, joblib_path):
        with open(joblib_path, "rb") as f:
            raw = f.read()
        self.scale_ = np.frombuffer(raw[784 : 784 + 120], dtype=np.float64).copy()
        self.min_ = np.frombuffer(raw[960 : 960 + 120], dtype=np.float64).copy()

    def transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.min_


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_COLS = (
    ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, "model_lstm.keras")
    scaler_path = os.path.join(BASE_DIR, "scaler.joblib")
    features_path = os.path.join(BASE_DIR, "feature_cols.joblib")

    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            model = NumpyLSTMModel(model_path)

    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = NumpyMinMaxScaler(scaler_path)

    try:
        features = joblib.load(features_path)
    except Exception:
        with open(features_path, "rb") as f:
            features = pickle.load(f)

    return model, scaler, features


model, scaler, features = load_assets()


def read_engine_dataframe(file_or_path):
    """Reads NASA C-MAPSS data whether it has headers or is raw space-separated numbers."""
    df_peek = pd.read_csv(file_or_path, sep=r"\s+", nrows=2)
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    if "sensor_2" in df_peek.columns or "unit_number" in df_peek.columns:
        return pd.read_csv(file_or_path, sep=r"\s+")

    df = pd.read_csv(file_or_path, sep=r"\s+", header=None)
    if df.shape[1] >= len(ALL_COLS):
        df = df.iloc[:, : len(ALL_COLS)]
        df.columns = ALL_COLS
    elif df.shape[1] == len(features):
        df.columns = features
    else:
        raise ValueError(
            f"Expected {len(ALL_COLS)} columns (raw NASA C-MAPSS format) or "
            f"{len(features)} sensor columns, but got {df.shape[1]} columns."
        )
    return df


st.title("Jet Engine Predictive Maintenance System")
st.markdown(
    "Predicting **Remaining Useful Life (RUL)** of turbofan engines using a Deep Learning LSTM network."
)

st.sidebar.header("Simulation Controls")
input_method = st.sidebar.radio(
    "Select Data Input Method:", ["Simulate Healthy Engine", "Upload Sensor CSV/TXT"]
)

if input_method == "Simulate Healthy Engine":
    st.subheader("Live Telemetry Simulation (Engine ID: #104)")
    st.write("Adjust key critical sensors to see real-time impact on Remaining Useful Life.")

    col1, col2, col3 = st.columns(3)
    with col1:
        s2 = st.slider("Total Temp at LPC Outlet (Sensor 2)", 640.0, 645.0, 642.5)
        s7 = st.slider("Total Pressure at HPC Outlet (Sensor 7)", 550.0, 556.0, 553.0)
    with col2:
        s11 = st.slider("Static Pressure at HPC Outlet (Sensor 11)", 46.0, 49.0, 47.5)
        s12 = st.slider("Ratio of Fuel Flow to Ps30 (Sensor 12)", 518.0, 523.0, 521.0)
    with col3:
        s15 = st.slider("Bypass Ratio (Sensor 15)", 8.3, 8.6, 8.4)

    base_data = np.full((30, len(features)), 0.5, dtype=np.float32)

    base_data[:, features.index("sensor_2")] = (s2 - 640.0) / 5.0
    base_data[:, features.index("sensor_7")] = (s7 - 550.0) / 6.0
    base_data[:, features.index("sensor_11")] = (s11 - 46.0) / 3.0
    base_data[:, features.index("sensor_12")] = (s12 - 518.0) / 5.0
    base_data[:, features.index("sensor_15")] = (s15 - 8.3) / 0.3

    sequence = np.array([base_data], dtype=np.float32)

    if st.button("Predict Engine Health", type="primary"):
        with st.spinner("Analyzing 30-cycle time-series window..."):
            predicted_rul = float(model.predict(sequence, verbose=0)[0][0])

            st.divider()
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="Predicted Remaining Useful Life (RUL)",
                    value=f"{int(predicted_rul)} Cycles",
                )

            with res_col2:
                if predicted_rul > 50:
                    st.success("STATUS: HEALTHY. Normal operations permitted.")
                elif predicted_rul > 20:
                    st.warning("STATUS: WARNING. Schedule maintenance within next 15 cycles.")
                else:
                    st.error("STATUS: CRITICAL! Ground aircraft immediately. Imminent failure.")

else:
    st.subheader("Batch Telemetry Upload")
    st.info("Upload a space-separated TXT or CSV file containing at least 30 rows of sensor data.")
    uploaded_file = st.file_uploader("Upload Engine Data", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            df = read_engine_dataframe(uploaded_file)

            if "unit_number" in df.columns and df["unit_number"].nunique() > 1:
                engine_ids = sorted(df["unit_number"].unique())
                selected_engine = st.selectbox("Select Engine ID (unit_number):", engine_ids)
                df_engine = df[df["unit_number"] == selected_engine].copy()
            else:
                df_engine = df.copy()

            st.write(f"Loaded **{len(df_engine)}** cycles of telemetry data. Preview (last 5 cycles):")
            st.dataframe(df_engine.tail(5), use_container_width=True)

            if len(df_engine) < 30:
                st.error(
                    f"Selected engine only has {len(df_engine)} rows. The LSTM model requires at least 30 cycles."
                )
            else:
                scaled_features = scaler.transform(df_engine[features].values)
                seq_array = np.array([scaled_features[-30:]], dtype=np.float32)
                pred = float(model.predict(seq_array, verbose=0)[0][0])

                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Predicted RUL", f"{int(pred)} Cycles")
                with col_b:
                    if pred > 50:
                        st.success("STATUS: HEALTHY. Normal operations permitted.")
                    elif pred > 20:
                        st.warning("STATUS: WARNING. Schedule maintenance within next 15 cycles.")
                    else:
                        st.error("STATUS: CRITICAL! Ground aircraft immediately. Imminent failure.")
        except Exception as e:
            st.error(f"Error processing file: {e}")