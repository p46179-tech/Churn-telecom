# app.py
# Streamlit Web Application for Customer Churn Prediction
# Modeled after interactive ML deployment structures[cite: 1].

import streamlit as st
import pandas as pd
import joblib

# Load trained classification model
filename = 'churn_model.sav'
try:
    loaded_model = joblib.load(open(filename, 'rb'))
except FileNotFoundError:
    loaded_model = None

# Define feature dataset column names
columns = [
    'Tenure_Months', 
    'Monthly_Charges', 
    'Total_Charges', 
    'Contract_Type', 
    'Tech_Support', 
    'Online_Security', 
    'Paperless_Billing', 
    'Num_Support_Tickets'
]

def predict_churn(features):
    """
    Predicts customer churn status based on provided feature inputs[cite: 1].
    """
    prediction = loaded_model.predict(features)
    return prediction

# Streamlit UI Setup
st.title("Customer Churn Prediction Dashboard")
st.write("Input customer metrics below to evaluate churn probability.")

# Input fields
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=120, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
contract_type = st.selectbox("Contract Type (0: Month-to-Month, 1: One Year, 2: Two Year)", [0, 1, 2])
tech_support = st.selectbox("Tech Support Included? (0: No, 1: Yes)", [0, 1])
online_security = st.selectbox("Online Security Included? (0: No, 1: Yes)", [0, 1])
paperless_billing = st.selectbox("Paperless Billing? (0: No, 1: Yes)", [0, 1])
support_tickets = st.number_input("Number of Support Tickets", min_value=0, max_value=20, value=1)

# Format inputs into dataframe[cite: 1]
input_data = pd.DataFrame([[
    tenure, monthly_charges, total_charges, contract_type,
    tech_support, online_security, paperless_billing, support_tickets
]], columns=columns)

# Run Prediction
if st.button("Predict Churn Risk"):
    if loaded_model is not None:
        prediction = predict_churn(input_data)
        if prediction[0] == 0:
            st.success("Predicted Churn Status: 0 (Customer Retained)")
        else:
            st.error("Predicted Churn Status: 1 (High Churn Risk)")
    else:
        st.warning("Model file 'churn_model.sav' not found. Please upload or train the model first.")