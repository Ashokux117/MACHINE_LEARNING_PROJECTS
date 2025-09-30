import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

st.title("KNN Regression Visualizer")

# Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Select columns for X and Y
    x_col = st.selectbox("Select Independent Variable (X)", data.columns)
    y_col = st.selectbox("Select Dependent Variable (Y)", data.columns)

    x = data[[x_col]].values
    y = data[y_col].values

    # Select k neighbors
    k = st.slider("Select number of neighbors (k)", 2, 6, 3)

    # Train KNN model
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(x, y)

    # Create smooth curve
    x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="red", label="Data Points")
    ax.plot(x_grid, knn_model.predict(x_grid), color="blue", label=f"KNN (k={k})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    st.pyplot(fig)

    # Prediction input
    input_val = st.number_input(f"Enter a value of {x_col} to predict:", min_value=float(min(x)), max_value=float(max(x)))
    if input_val:
        pred = knn_model.predict([[input_val]])[0]
        st.success(f" Predicted {y_col} for {x_col}={input_val}: **{pred:.2f}**")
