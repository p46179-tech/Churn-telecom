import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer information to predict churn risk.")

MODEL_FILE = "churn_model.sav"

# Load the trained pipeline
if not os.path.exists(MODEL_FILE):
    st.error("churn_model.sav was not found in the GitHub repository.")
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error("The model could not be loaded.")
    st.error(
        "Make sure Streamlit is using scikit-learn 1.6.1, "
        "the same version used to create churn_model.sav."
    )
    st.exception(e)
    st.stop()

st.subheader("Customer Information")

call_failure = st.number_input("Call Failure", min_value=0, value=5)
complains = st.selectbox("Complains", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
subscription_length = st.number_input("Subscription Length", min_value=0, value=12)
charge_amount = st.number_input("Charge Amount", min_value=0, value=30)
seconds_of_use = st.number_input("Seconds of Use", min_value=0, value=1000)
frequency_use = st.number_input("Frequency of Use", min_value=0, value=20)
frequency_sms = st.number_input("Frequency of SMS", min_value=0, value=10)
distinct_called_numbers = st.number_input("Distinct Called Numbers", min_value=0, value=10)

age_group = st.selectbox("Age Group", [1, 2, 3, 4, 5])
tariff_plan = st.selectbox("Tariff Plan", [1, 2])
status = st.selectbox("Status", [1, 2])

age = st.number_input("Age", min_value=0, value=30)
customer_value = st.number_input("Customer Value", min_value=0.0, value=100.0)

input_data = pd.DataFrame([{
    "Call  Failure": call_failure,
    "Complains": complains,
    "Subscription  Length": subscription_length,
    "Charge  Amount": charge_amount,
    "Seconds of Use": seconds_of_use,
    "Frequency of use": frequency_use,
    "Frequency of SMS": frequency_sms,
    "Distinct Called Numbers": distinct_called_numbers,
    "Age Group": age_group,
    "Tariff Plan": tariff_plan,
    "Status": status,
    "Age": age,
    "Customer Value": customer_value
}])

if st.button("Predict Churn Risk"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Churn Risk")
        else:
            st.success("✅ Customer is likely to be retained")

        st.metric("Churn Probability", f"{probability:.2%}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
