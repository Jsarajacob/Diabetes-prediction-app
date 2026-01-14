# -*- coding: utf-8 -*-


import streamlit as st
import numpy as np
import joblib


model = joblib.load("diabetes_model.pkl")


st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter clinical details to predict diabetes status:")


glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, key="glucose")

bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, key="bp")

insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, key="insulin")

bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, key="bmi")

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, key="dpf")

age = st.number_input("Age", min_value=1, max_value=120, value=33, key="age")


input_data = np.array([[glucose, bp, insulin, bmi, dpf, age]])
input_final = input_data




if st.button("Predict"):
    probability = model.predict_proba(input_scaled)[0][1]

    # Adjustable threshold
    threshold = 0.5

    st.write(f"Risk Probability: {probability:.4f}")

    if probability >= threshold:
        st.error(f"🟥 **Diabetes: YES**\n\nRisk Probability: {probability:.2f}")
    else:
        st.success(f"🟩 **Diabetes: NO**\n\nRisk Probability: {probability:.2f}")
  
