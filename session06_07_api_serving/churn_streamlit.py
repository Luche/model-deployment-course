"""
Session 06-07 – Streamlit Frontend: Calls Churn FastAPI endpoint
Run AFTER starting the FastAPI server:
  uvicorn churn_fastapi:app --reload
Then run: streamlit run churn_streamlit.py
"""

import streamlit as st
import requests


def main():
    st.title("Customer Churn Prediction – API Demo")

    age              = st.number_input("Age",                           0, 100)
    gender           = st.radio("Gender",                               ["Male", "Female"])
    tenure           = st.number_input("Tenure (months)",               0, 100)
    usage_freq       = st.number_input("Usage Frequency (times/month)", 0, 100)
    support_call     = st.number_input("Support Calls",                 0, 10)
    payment_delay    = st.number_input("Payment Delay (days)",          0, 30)
    subs_type        = st.radio("Subscription Type",                    ["Standard", "Premium", "Basic"])
    contract_length  = st.radio("Contract Length",                      ["Annual", "Quarterly", "Monthly"])
    total_spend      = st.number_input("Total Spend",                   0, 1_000_000)
    last_interaction = st.number_input("Last Interaction (days ago)",   0, 30)

    data = {
        "Age":              int(age),
        "Gender":           gender,
        "Tenure":           int(tenure),
        "UsageFrequency":   int(usage_freq),
        "SupportCalls":     int(support_call),
        "PaymentDelay":     int(payment_delay),
        "SubscriptionType": subs_type,
        "ContractLength":   contract_length,
        "TotalSpend":       int(total_spend),
        "LastInteraction":  int(last_interaction),
    }

    if st.button("Make Prediction"):
        result = make_prediction(data)
        if result is not None:
            label = "Will Churn" if result == 1 else "Will NOT Churn"
            st.success(f"Churn Prediction: {label}")


def make_prediction(features):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=features, timeout=5)
        return response.json()["prediction"]
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server. Start it first:\n"
                 "`uvicorn churn_fastapi:app --reload`\n\n"
                 "Or use the standalone version: `streamlit run churn_streamlit_standalone.py`")
        return None


if __name__ == "__main__":
    main()
