# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RmhZn3N2C6cTTpbZ2unuTcP5Iolk9sfr
"""

import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("titanic_survival_model.pkl")

# Streamlit App UI
st.title("🚢 Titanic Survival Predictor")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0, value=30)
sex = st.radio("Sex", ["Male", "Female"])
embarked = st.radio("Embarked", ["C", "Q", "S"])
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)

# Convert Inputs to Match Model Features
sex = 1 if sex == "Male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create DataFrame for Prediction
input_data = pd.DataFrame([[pclass, age, fare, sex, embarked_C, embarked_Q, embarked_S, sibsp, parch]],
                          columns=["Pclass", "Age", "Fare", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S", "SibSp", "Parch"])

# Make Prediction
prediction = model.predict(input_data)[0]

# Display Result
if prediction == 1:
    st.success("✅ This passenger would have SURVIVED!")
else:
    st.error("❌ This passenger would NOT have survived.")

import os
import joblib

# Check if the model file exists before loading
model_path = "titanic_survival_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)  # Load the model only if it exists

