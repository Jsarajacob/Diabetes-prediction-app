import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter clinical details to predict diabetes risk:")

# ---------- Inputs ----------
glucose = st.number_input(
    "Glucose (mg/dL)", min_value=0, max_value=300, value=120
)

blood_pressure = st.number_input(
    "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70
)

insulin = st.number_input(
    "Insulin (µU/mL)", min_value=0, max_value=900, value=80
)

bmi = st.number_input(
    "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)

age = st.number_input(
    "Age (years)", min_value=1, max_value=120, value=33
)

# ---------- Prediction ----------
if st.button("Predict"):
    input_data = np.array([[glucose, blood_pressure, insulin, bmi, dpf, age]])

    probability = model.predict_proba(input_data)[0][1]

    st.write(f"### Risk Probability: {probability:.3f}")

    if probability >= 0.5:
        st.error(f"🟥 **Diabetes: YES**\n\nRisk: {probability*100:.1f}%")
    else:
        st.success(f"🟩 **Diabetes: NO**\n\nRisk: {probability*100:.1f}%")

st.caption("⚠️ Educational use only. Not a medical diagnosis.")
