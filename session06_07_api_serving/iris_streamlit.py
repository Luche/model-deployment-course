"""
Session 06-07 – Streamlit Frontend: Calls Iris FastAPI endpoint
Run AFTER starting the FastAPI server:
  uvicorn iris_fastapi:app --reload
Then run: streamlit run iris_streamlit.py
"""

import streamlit as st
import requests


def main():
    st.title("Machine Learning Model Deployment – Iris")

    sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width  = st.slider("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width  = st.slider("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2)

    if st.button("Make Prediction"):
        features = {
            "sepal_length": sepal_length,
            "sepal_width":  sepal_width,
            "petal_length": petal_length,
            "petal_width":  petal_width,
        }
        result = make_prediction(features)
        if result is not None:
            st.success(f"Predicted Species: {result}")


def make_prediction(features):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=features, timeout=5)
        return response.json()['prediction']
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server. Start it first:\n"
                 "`uvicorn iris_fastapi:app --reload`\n\n"
                 "Or use the standalone version: `streamlit run iris_streamlit_standalone.py`")
        return None


if __name__ == "__main__":
    main()
