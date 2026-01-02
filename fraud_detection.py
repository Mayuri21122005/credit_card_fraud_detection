# Streamlit App: Credit Card Fraud Detection

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Title Section
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>💳 Credit Card Fraud Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Machine Learning based real-time fraud detection system</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📊 Dataset Info")
st.sidebar.write("Total Transactions:", data.shape[0])
st.sidebar.write("Fraud Cases:", data['Class'].sum())
st.sidebar.write("Normal Cases:", data.shape[0] - data['Class'].sum())

# -----------------------------
# Data Preparation
# -----------------------------
X = data.drop('Class', axis=1)
y = data['Class']

# Scale Amount column
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Improved Model
# -----------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# -----------------------------
# Model Evaluation Section
# -----------------------------
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{model.score(X_test, y_test)*100:.2f}%")

with col2:
    st.metric("Fraud Detection Recall", 
              f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']*100:.2f}%")

# -----------------------------
# User Input Section
# -----------------------------
st.divider()
st.subheader("🧪 Test a Transaction")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

if st.button("🔍 Predict Transaction"):
    # Take a random transaction as base
    sample = X_test.sample(1).copy()
    sample['Amount'] = scaler.transform([[amount]])

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("⚠️ FRAUD Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)

