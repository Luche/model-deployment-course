"""
Session 05 – Preprocessing (No-Pipeline approach)
Manual imputation and ordinal encoding steps, saved as artifacts.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def load_data():
    os.makedirs("artifacts", exist_ok=True)
    df = pd.read_csv("ingested/customer_churn.csv", sep=";")

    X = df.drop(['Churn', 'CustomerID'], axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def missing_value(x_train, x_test):
    impute_stats = {
        "tenure_mean":        x_train['Tenure'].mean(),
        "support_calls_mean": x_train['Support Calls'].mean(),
        "total_spend_mean":   x_train['Total Spend'].mean(),
        "gender_mode":        x_train['Gender'].mode()[0],
    }

    for df in [x_train, x_test]:
        df['Tenure']        = df['Tenure'].fillna(impute_stats['tenure_mean'])
        df['Support Calls'] = df['Support Calls'].fillna(impute_stats['support_calls_mean'])
        df['Total Spend']   = df['Total Spend'].fillna(impute_stats['total_spend_mean'])
        df['Gender']        = df['Gender'].fillna(impute_stats['gender_mode'])

    joblib.dump(impute_stats, "artifacts/impute_stats.pkl")
    return x_train, x_test


def encoder(x_train, x_test):
    x_train = x_train.replace({"Gender": {"Male": 1, "Female": 0}})
    x_test  = x_test.replace({"Gender": {"Male": 1, "Female": 0}})

    subs_categories = [['Basic', 'Standard', 'Premium']]
    cont_categories = [['Monthly', 'Quarterly', 'Annual']]

    subs_encoder = OrdinalEncoder(categories=subs_categories,
                                   handle_unknown='use_encoded_value', unknown_value=-1)
    cont_encoder = OrdinalEncoder(categories=cont_categories,
                                   handle_unknown='use_encoded_value', unknown_value=-1)

    for (enc, col, new_col, tr, te) in [
        (subs_encoder, 'Subscription Type', 'Subscription Type Ordinal', x_train, x_test),
        (cont_encoder, 'Contract Length',   'Contract Length Ordinal',   x_train, x_test),
    ]:
        tr_enc = pd.DataFrame(enc.fit_transform(tr[[col]]), columns=[new_col])
        te_enc = pd.DataFrame(enc.transform(te[[col]]),     columns=[new_col])
        x_train = pd.concat([x_train.reset_index(drop=True), tr_enc], axis=1)
        x_test  = pd.concat([x_test.reset_index(drop=True),  te_enc], axis=1)

    x_train = x_train.drop(['Subscription Type', 'Contract Length'], axis=1)
    x_test  = x_test.drop(['Subscription Type', 'Contract Length'], axis=1)

    joblib.dump(subs_encoder, 'artifacts/ordinal_encode_subs.pkl')
    joblib.dump(cont_encoder, 'artifacts/ordinal_encode_cont.pkl')

    return x_train, x_test
