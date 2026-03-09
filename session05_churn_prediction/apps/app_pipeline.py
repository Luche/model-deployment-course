import pandas as pd
import streamlit as st

from config.config import ARTIFACT_PIPELINE
from src.utils.io import load_artifact


@st.cache_resource
def load_pipeline():
    try:
        return load_artifact(ARTIFACT_PIPELINE)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


def main():
    st.title("Customer Churn Prediction (Pipeline)")

    model = load_pipeline()

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
    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        if prediction == 1:
            st.error("Churn Prediction: Will Churn")
        else:
            st.success("Churn Prediction: Will NOT Churn")


if __name__ == "__main__":
    main()
