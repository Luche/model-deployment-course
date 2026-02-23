"""
Session 04 – Streamlit App
Serve the trained Iris Random Forest classifier via a web UI.
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load preprocessor and model
scaler = joblib.load("artifacts/preprocessor.pkl")
model  = joblib.load("artifacts/model.pkl")


def main():
    st.title("Machine Learning Iris Prediction")

    sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width  = st.slider("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width  = st.slider("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2)

    if st.button("Make Prediction"):
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        features = pd.DataFrame(features, columns=['sepal_length', 'sepal_width',
                                               'petal_length', 'petal_width'])
        result = make_prediction(features)
        st.success(f"Predicted Species: {result}")


def make_prediction(features):
    X_scaled = scaler.transform(features)
    prediction = model.predict(X_scaled)
    return prediction[0]


if __name__ == "__main__":
    main()
