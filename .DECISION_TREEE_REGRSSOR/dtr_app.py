# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from io import StringIO

# Streamlit page setup
st.set_page_config(page_title="Decision Tree Regression App", layout="centered")

st.title("Decision Tree Regression - Salary Prediction")
st.write("Upload your dataset and visualize Decision Tree Regression results interactively.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file (e.g., emp_sal.csv)", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    Data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(Data.head())

    # Select columns for features and target
    feature_col = st.selectbox("Select feature column (X)", Data.columns)
    target_col = st.selectbox("Select target column (Y)", Data.columns)

    x = Data[[feature_col]].values
    y = Data[target_col].values

    # Train Decision Tree model
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x, y)

    # User input for prediction
    st.subheader("Make a Prediction")
    input_value = st.number_input("Enter a value for prediction:", min_value=float(np.min(x)), max_value=float(np.max(x)), value=float(np.median(x)))
    pred = model.predict([[input_value]])

    st.success(f"Predicted Output for {input_value}: **{pred[0]:.2f}**")

    # Plotting
    st.subheader("Model Visualization")
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', label='Actual Data')
    ax.plot(x, model.predict(x), color='blue', label='Model Prediction')
    ax.set_title('Decision Tree Regression')
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("👆 Please upload a CSV file to continue.")

