"""
Session 05 – Streamlit App (No-Pipeline approach)
Loads manual preprocessing artifacts and model separately.
Run with: streamlit run app_churnNoPipeline.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd

model             = joblib.load('artifacts/model_churnNoPipeline.pkl')
impute_stats      = joblib.load('artifacts/impute_stats.pkl')
ordinal_enc_subs  = joblib.load('artifacts/ordinal_encode_subs.pkl')
ordinal_enc_cont  = joblib.load('artifacts/ordinal_encode_cont.pkl')


def main():
    st.title("Customer Churn Prediction (No-Pipeline)")

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
        'Age': int(age), 'Gender': gender, 'Tenure': int(tenure),
        'Usage Frequency': int(usage_freq), 'Support Calls': int(support_call),
        'Payment Delay': int(payment_delay), 'Subscription Type': subs_type,
        'Contract Length': contract_length, 'Total Spend': int(total_spend),
        'Last Interaction': int(last_interaction),
    }
    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
    df['Gender'] = df['Gender'].map({"Male": 1, "Female": 0})

    for col, key in [('Tenure', 'tenure_mean'), ('Support Calls', 'support_calls_mean'),
                      ('Total Spend', 'total_spend_mean'), ('Gender', 'gender_mode')]:
        df[col] = df[col].fillna(impute_stats[key])

    subs_enc = pd.DataFrame(ordinal_enc_subs.transform(df[['Subscription Type']]),
                             columns=['Subscription Type Ordinal'])
    cont_enc = pd.DataFrame(ordinal_enc_cont.transform(df[['Contract Length']]),
                             columns=['Contract Length Ordinal'])
    df = pd.concat([df, subs_enc, cont_enc], axis=1)
    df = df.drop(['Subscription Type', 'Contract Length'], axis=1)

    if st.button("Make Prediction"):
        prediction = model.predict(np.array(df).reshape(1, -1))
        st.success(f"Churn Prediction: {'Will Churn' if prediction[0] == 1 else 'Will NOT Churn'}")


if __name__ == "__main__":
    main()
