import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter clinical details to predict diabetes risk:")

# ---------- Inputs ----------
glucose = st.text_input("Glucose", placeholder="mg/dL (e.g., 120)")
bp = st.text_input("Blood Pressure", placeholder="Diastolic BP (e.g., 70)")
insulin = st.text_input("Insulin", placeholder="µU/mL (e.g., 80)")
bmi = st.text_input("BMI", placeholder="kg/m² (e.g., 25.5)")
dpf = st.text_input("Diabetes Pedigree Function", placeholder="e.g., 0.52")
age = st.text_input("Age", placeholder="years (e.g., 33)")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        # Convert to numeric
        input_data = np.array([[ 
            float(glucose),
            float(bp),
            float(insulin),
            float(bmi),
            float(dpf),
            float(age)
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"🟥 **Diabetes: YES**\n\nRisk Probability: **{probability:.2f}**")
        else:
            st.success(f"🟩 **Diabetes: NO**\n\nRisk Probability: **{probability:.2f}**")

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values in all fields")

st.caption("⚠️ Educational use only. Not a medical diagnosis.")
