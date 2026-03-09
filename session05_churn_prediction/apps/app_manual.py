import pandas as pd
import streamlit as st

from config.config import (
    NUM_FEATURES, CAT_FEATURES,
    ARTIFACT_MODEL_MANUAL, ARTIFACT_NUM_IMPUTER, ARTIFACT_CAT_IMPUTER, ARTIFACT_CAT_ENCODER,
)
from src.utils.io import load_manual_artifacts


@st.cache_resource
def load_artifacts():
    try:
        return load_manual_artifacts(
            ARTIFACT_MODEL_MANUAL, ARTIFACT_NUM_IMPUTER, ARTIFACT_CAT_IMPUTER, ARTIFACT_CAT_ENCODER,
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


def main():
    st.title("Customer Churn Prediction (No-Pipeline)")

    model, num_imputer, cat_imputer, cat_encoder = load_artifacts()

    age              = st.number_input("Age",                              0, 100)
    gender           = st.radio("Gender",                                  ["Male", "Female"])
    tenure           = st.number_input("Tenure (months)",                  0, 100)
    usage_freq       = st.number_input("Usage Frequency (times/month)",    0, 100)
    support_call     = st.number_input("Support Calls",                    0, 10)
    payment_delay    = st.number_input("Payment Delay (days)",             0, 30)
    subs_type        = st.radio("Subscription Type",                       ["Standard", "Premium", "Basic"])
    contract_length  = st.radio("Contract Length",                         ["Annual", "Quarterly", "Monthly"])
    total_spend      = st.number_input("Total Spend",                      0, 1_000_000)
    last_interaction = st.number_input("Last Interaction (days ago)",      0, 30)

    data = {
        "Age":               int(age),
        "Gender":            gender,
        "Tenure":            int(tenure),
        "Usage Frequency":   int(usage_freq),
        "Support Calls":     int(support_call),
        "Payment Delay":     int(payment_delay),
        "Subscription Type": subs_type,
        "Contract Length":   contract_length,
        "Total Spend":       int(total_spend),
        "Last Interaction":  int(last_interaction),
    }
    df = pd.DataFrame([data])

    df[NUM_FEATURES] = num_imputer.transform(df[NUM_FEATURES])
    df[CAT_FEATURES] = cat_imputer.transform(df[CAT_FEATURES])
    df[CAT_FEATURES] = cat_encoder.transform(df[CAT_FEATURES])

    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        if prediction == 1:
            st.error("Churn Prediction: Will Churn")
        else:
            st.success("Churn Prediction: Will NOT Churn")


if __name__ == "__main__":
    main()
