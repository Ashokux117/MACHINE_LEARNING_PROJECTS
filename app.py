import streamlit as st
import pickle 
import numpy as np 

# Load trained model
with open(r"C:\Users\DELL\Notes_and_practice\MACHINE_LEARNING\LINEAR_REGRESSION_MODEL\lrg_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("Salary Prediction App ")

# Description
st.write("This app predicts the salary based on years of experience using a simple Linear Regression model.")

# Input field
years_experience = st.number_input(
    "Enter years of experience:", 
    min_value=0.0, 
    max_value=50.0, 
    step=0.1
)

# Predict button
if st.button("Predict Salary"):
    # Reshape input for model
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)
    
    # Show result
    st.success(f"Predicted Salary: Rs {prediction[0]:,.2f}")
