import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -------------------------------
# Custom CSS (BACKGROUND + UI)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(
            rgba(0, 0, 0, 0.75),
            rgba(0, 0, 0, 0.75)
        ),
        url("https://images.unsplash.com/photo-1614064641938-3bbee52942c7");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }

    h1, h2, h3 {
        color: #ff4b4b;
        text-align: center;
    }

    .glass {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
    }

    .metric-box {
        background: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1>💳 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>AI-powered system to identify suspicious transactions in real time</p>",
    unsafe_allow_html=True
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.header("📊 Dataset Summary")
st.sidebar.write("Total Transactions:", data.shape[0])
st.sidebar.write("Fraud Cases:", data['Class'].sum())
st.sidebar.write("Normal Transactions:", data.shape[0] - data['Class'].sum())

# -------------------------------
# Data Preparation
# -------------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# -------------------------------
# Metrics Section
# -------------------------------
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"<div class='metric-box'>Accuracy<br><b>{model.score(X_test, y_test)*100:.2f}%</b></div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div class='metric-box'>Fraud Recall<br><b>{report['1']['recall']*100:.2f}%</b></div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("🔍 Test a Transaction")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=500.0)

if st.button("🚨 Analyze Transaction"):
    sample = X.iloc[0].values.reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    if prediction[0] == 1:
        st.error("⚠️ FRAUD DETECTED — Transaction Blocked")
    else:
        st.success("✅ Transaction is SAFE")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
