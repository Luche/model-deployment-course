"""
Session 05 – Streamlit App (sklearn Pipeline approach)
Single pipeline artifact handles all preprocessing + prediction.
Run with: streamlit run app_churnPipeline.py
"""

import streamlit as st
import joblib
import pandas as pd

model = joblib.load('artifacts/churn_prediction_pipeline.pkl')


def main():
    st.title("Customer Churn Prediction (Pipeline)")

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
        'Age': int(age), 'Gender': gender, 'Tenure': int(tenure),
        'UsageFrequency': int(usage_freq), 'SupportCalls': int(support_call),
        'PaymentDelay': int(payment_delay), 'SubscriptionType': subs_type,
        'ContractLength': contract_length, 'TotalSpend': int(total_spend),
        'LastInteraction': int(last_interaction),
    }
    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        st.success(f"Churn Prediction: {'Will Churn' if prediction == 1 else 'Will NOT Churn'}")


if __name__ == "__main__":
    main()
