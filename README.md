# 💳 Credit Card Fraud Detection using Machine Learning

An end-to-end **machine learning–based credit card fraud detection system** built using **Python, Scikit-Learn, and Streamlit**.  
This project detects whether a transaction is **Fraudulent or Legitimate** based on learned transaction behavior patterns and provides an **interactive web interface** for real-time testing.

---

## 🚀 Project Overview

Credit card fraud is a major real-world problem faced by banks and financial institutions.  
This project uses **Logistic Regression** to classify transactions as fraud or safe using a **real, highly imbalanced dataset**.

Key highlights:
- Uses a **real Kaggle dataset** with 284,807 transactions
- Handles **class imbalance** using class weighting
- Uses **feature scaling** for better model convergence
- Deployed as an **interactive Streamlit web app**
- Includes a **fraud simulation toggle** for demonstration

---

## 🧠 How the System Works

1. The dataset is loaded and preprocessed
2. Transaction features are scaled using `StandardScaler`
3. A Logistic Regression model is trained with balanced class weights
4. The user inputs a transaction amount
5. The system internally simulates transaction behavior
6. The model predicts whether the transaction is **Fraud** or **Safe**
7. The result is displayed instantly on the web interface

> ⚠️ Fraud detection is based on **behavioral patterns**, not transaction amount alone.

---

## 📊 Dataset Details

- Source: Kaggle (Credit Card Fraud Detection – ULB)
- Total Transactions: **284,807**
- Fraud Cases: **492**
- Normal Transactions: **284,315**
- Features:
  - `Time`
  - `V1` to `V28` (PCA-transformed behavioral features)
  - `Amount`
  - `Class` (Target: 0 = Safe, 1 = Fraud)

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Machine Learning (Logistic Regression)

---

## 🧪 Key Machine Learning Techniques

- Binary Classification
- Handling Imbalanced Data (`class_weight='balanced'`)
- Feature Scaling (`StandardScaler`)
- Train–Test Split with Stratification
- Real-world evaluation metrics (Precision, Recall)

---

## 🎛️ Fraud Simulation Toggle (Important Feature)

The application includes a **“Simulate Suspicious Behavior” toggle**:

- **OFF** → Tests a normal transaction pattern
- **ON** → Tests a real fraud transaction pattern from the dataset

This allows realistic demonstration of fraud detection without manually entering encrypted features.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Mayuri211222005/credit_card_fraud_detection.git
cd credit_card_fraud_detection
