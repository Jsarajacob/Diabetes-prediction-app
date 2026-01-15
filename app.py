import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter clinical details to predict diabetes risk:")

# ---------- Inputs ----------
glucose = st.text_input("Glucose", placeholder="mg/dL (50-500)")
bp = st.text_input("Blood Pressure", placeholder="Diastolic BP (40-200)")
insulin = st.text_input("Insulin", placeholder="µU/mL (15-900)")
bmi = st.text_input("BMI", placeholder="kg/m² (15-70)")
dpf = st.text_input("Diabetes Pedigree Function", placeholder="risk score (0.0-3.0)")
age = st.text_input("Age", placeholder="years (5-95)")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        # Convert inputs to float
        input_data = np.array([[ 
            float(glucose),
            float(bp),
            float(insulin),
            float(bmi),
            float(dpf),
            float(age)
        ]])

        # Predict probability
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Result")
        st.write(f"### Risk Probability: {probability:.3f}")

        # Decision threshold
        threshold = 0.5

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

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values in all fields")

st.caption("⚠️ Educational use only. Not a medical diagnosis.")
