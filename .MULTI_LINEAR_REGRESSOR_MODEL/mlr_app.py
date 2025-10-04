import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# ----------------------------
# Title
# ----------------------------
st.title("Multiple Linear Regression App")
st.write("Upload your Investment dataset and make predictions easily!")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", data.head())

    # Separate features and target
    x = data.iloc[:, :-1]
    y = data.iloc[:, 4]

    # Convert categorical columns to dummy variables
    x = pd.get_dummies(x, dtype=int)

    # Train-test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Model evaluation
    bias = regressor.score(x_train, y_train)
    variance = regressor.score(x_test, y_test)

    st.write("### Model Training Complete")
    st.write(f"**Bias (Train Score):** {bias:.4f}")
    st.write(f"**Variance (Test Score):** {variance:.4f}")
    st.write(f"**Intercept:** {regressor.intercept_:.4f}")

    # ----------------------------
    # Prediction Section
    # ----------------------------
    st.write("### Make a Prediction")

    input_data = []
    for col in x.columns:
        val = st.number_input(f"Enter value for {col}:", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = regressor.predict(input_array)
        st.success(f"💰 Predicted Output: {prediction[0]:.2f}")

else:
    st.info("Please upload your dataset first.")

# ----------------------------
# Run using:
# streamlit run app.py
# ----------------------------
