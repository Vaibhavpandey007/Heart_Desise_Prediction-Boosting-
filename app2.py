import streamlit as st
import pandas as pd
import joblib

#Loading data
model = joblib.load("ADABoost_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction by Vaibhav")
st.markdown("Provide your details to check the heart disease prediction: ")

age = st.slider("Age", 18, 100, 40)
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol Level", 85, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
max_hr = st.slider("Maximum Heart Rate", 60, 202, 150)
oldpeak = st.slider("Oldpeak (ST Depression)", -2.6, 6.2, 0.0)

sex_m = st.selectbox("Sex", ["Female", "Male"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

data = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,
    "Sex_M": 1 if sex_m == "Male" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0,
}

# Convert to DataFrame and align columns
input_df = pd.DataFrame([data])
input_df = input_df.reindex(columns=expected_columns, fill_value=0)


input_scaled = scaler.transform(input_df)

st.write("Input DataFrame before scaling:")
st.write(input_df)

st.write("Scaled input:")
st.write(input_scaled)

proba = model.predict_proba(input_scaled)[0]
st.write("Prediction probabilities:", proba)

#predict..
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High Chances of Heart Disease")
    else:
        st.success("✅ Less Chances , You are Safe")