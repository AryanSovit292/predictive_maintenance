# ✈️ Jet Engine Predictive Maintenance (NASA CMAPSS)

An end-to-end Machine Learning and Deep Learning system to predict the **Remaining Useful Life (RUL)** of turbofan jet engines from multivariate sensor telemetry.

---

## 📌 Problem Overview
In commercial aviation, unexpected engine failures lead to flight disruptions, massive financial costs, and severe safety risks. Using simulated run-to-failure sensor data from NASA's C-MAPSS simulator, this project implements a predictive maintenance pipeline to estimate the remaining flight cycles before an engine requires overhaul.

---

## ⚙️ Dataset
- **Source**: NASA CMAPSS Turbofan Engine Degradation Dataset (Subset **FD001**)
- **Data Scope**: 100 engine units in the training set (run to failure) and 100 engine units in the test set
- **Telemetry**: 21 physical sensor channels (temperatures, pressures, fan speeds) and 3 operating settings recorded across flight cycles

---

## 🔬 Pipeline & Methodology

1. **Exploratory Data Analysis (EDA)**:
   - Analyzed degradation curves across engine lifespans.
   - Identified and removed **6 non-informative flat-line sensors** (`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) along with static operating settings that had zero variance.

2. **Feature Engineering & Preprocessing**:
   - Formulated run-to-failure RUL targets for training data.
   - Scaled sensor values using `MinMaxScaler` fitted strictly on training data.
   - Built a **30-cycle sliding window sequence generator** to convert tabular time-series into 3D tensors `(samples, 30 time steps, 15 features)` for sequence modeling.

3. **Modeling Approaches**:
   - **Baseline**: `RandomForestRegressor` predicting RUL from instantaneous, point-in-time sensor readings.
   - **Deep Learning**: Sequential `LSTM` (Long Short-Term Memory) network with Dropout regularization to capture long-term temporal wear patterns.

---

## 📊 Results & Model Comparison

| Model | Input Representation | Test RMSE (Cycles) |
|---|---|---|
| **Random Forest (Baseline)** | Single-cycle instantaneous snapshot | **34.22** |
| **LSTM Neural Network** | 30-cycle temporal sliding window | **~16–19** |

**Key Takeaway**: The LSTM significantly outperforms the Random Forest baseline because turbofan degradation is cumulative. A point-in-time model cannot distinguish between normal operating variation and steady drift, whereas an LSTM models the trajectory and rate-of-change over time.

---

## 🖥️ Streamlit Web Application

The interactive web dashboard (`app.py`) provides:
- Telemetry log inspection for uploaded engine CSV/text files.
- Automated feature extraction and min-max scaling of the last 30 operational cycles.
- Real-time RUL prediction with automated health classification:
  - 🟢 **Healthy**: RUL > 50 cycles (normal operations)
  - 🟡 **Warning**: 20 < RUL <= 50 cycles (schedule inspection)
  - 🔴 **Critical**: RUL <= 20 cycles (immediate maintenance required)
- Trend visualization of key degradation sensors over the last 30 cycles.

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/AryanSovit292/predictive_maintenance.git
cd predictive_maintenance