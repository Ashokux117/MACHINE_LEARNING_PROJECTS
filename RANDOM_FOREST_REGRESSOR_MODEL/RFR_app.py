# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Streamlit page configuration
st.set_page_config(page_title="Random Forest Regression App", layout="centered")

# Title and description
st.title("Random Forest Regression - Salary Prediction App")
st.write("Upload your dataset, select feature and target columns, and predict salaries using Random Forest Regression.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file (e.g., emp_sal.csv)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    Data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(Data.head())

    # Feature and Target Selection
    feature_col = st.selectbox("Select feature column (X)", Data.columns)
    target_col = st.selectbox("Select target column (Y)", Data.columns)

    x = Data[[feature_col]].values
    y = Data[target_col].values

    # Slider for number of estimators
    n_estimators = st.slider("Select number of trees (n_estimators)", 10, 300, 100, step=10)

    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    model.fit(x, y)

    # Prediction input
    st.subheader("Make a Prediction")
    input_value = st.number_input(
        f"Enter a {feature_col} value for prediction:",
        min_value=float(np.min(x)),
        max_value=float(np.max(x)),
        value=float(np.median(x))
    )
    pred = model.predict([[input_value]])

    st.success(f"Predicted {target_col} for {input_value}: **{pred[0]:.2f}**")

    # Plot results
    st.subheader("Model Visualization")
    X_grid = np.arange(min(x), max(x), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', label='Actual Data')
    ax.plot(X_grid, model.predict(X_grid), color='green', label='Random Forest Prediction')
    ax.set_title('Random Forest Regression')
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to continue.")
