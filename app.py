# -*- coding: utf-8 -*-


import streamlit as st
import numpy as np
import joblib

# =========================
# Load trained Random Forest model
# =========================
model = joblib.load("diabetes_model.pkl")

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="centered"
)

st.title("Diabetes Risk Prediction")
st.write("Enter clinical details to predict diabetes risk:")

# =========================
# User Inputs (RAW VALUES)
# =========================
glucose = st.number_input(
    "Glucose (mg/dL)", min_value=0, max_value=300, value=120, key="glucose"
)

bp = st.number_input(
    "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, key="bp"
)

insulin = st.number_input(
    "Insulin (µU/mL)", min_value=0, max_value=900, value=80, key="insulin"
)

bmi = st.number_input(
    "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, key="bmi"
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0, value=0.5, key="dpf"
)

age = st.number_input(
    "Age (years)", min_value=1, max_value=120, value=33, key="age"
)

# =========================
# Prepare input (NO SCALING)
# Feature order MUST match training
# =========================
input_data = np.array([[glucose, bp, insulin, bmi, dpf, age]])

# =========================
# Prediction
# =========================
if st.button("Predict"):
    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.5  # clinical decision threshold

    st.write(f"### Risk Probability: {probability:.3f}")

    if probability >= threshold:
        st.error(
            f"🟥 **Diabetes: YES**\n\n"
            f"Estimated Risk: **{probability*100:.1f}%**"
        )
    else:
        st.success(
            f"🟩 **Diabetes: NO**\n\n"
            f"Estimated Risk: **{probability*100:.1f}%**"
        )
