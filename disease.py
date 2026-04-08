import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DB_PATH = "predictions.db"

st.set_page_config(layout="wide", page_title="Advanced Parkinson Dashboard")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result TEXT,
    probability REAL,
    time TEXT
)
""")
conn.commit()

@st.cache_resource
def train_model():
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    )

    X = df.drop(columns=["status", "name"])
    y = df["status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler, metrics, X, y, X_test, y_test, y_pred, y_prob


model, scaler, metrics, X, y, X_test, y_test, y_pred, y_prob = train_model()

st.title("🧠 Advanced Parkinson Disease Analytics Dashboard")

st.sidebar.header("🔎 Filters")

filter_result = st.sidebar.selectbox(
    "Filter by Prediction",
    ["All", "Healthy", "Parkinson's"]
)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Prediction", "📈 Analytics"])

with tab1:
    st.subheader("📊 Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    c2.metric("Precision", f"{metrics['precision']:.2f}")
    c3.metric("Recall", f"{metrics['recall']:.2f}")
    c4.metric("F1 Score", f"{metrics['f1']:.2f}")
    c5.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")

    st.subheader("📉 ROC Curve")
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr)
    st.pyplot(fig)

    st.subheader("📊 Confusion Matrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax2)
    st.pyplot(fig2)

with tab2:
    st.subheader("🔮 Predict Parkinson's")

    inputs = []
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        val = cols[i % 3].number_input(col, value=float(X[col].mean()), format="%.6g")
        inputs.append(val)

    if st.button("Predict"):
        arr = scaler.transform([inputs])
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0][pred]

        label = "Parkinson's" if pred == 1 else "Healthy"

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {prob*100:.2f}%")

        cursor.execute(
            "INSERT INTO predictions (result, probability, time) VALUES (?, ?, ?)",
            (label, float(prob), str(datetime.now()))
        )
        conn.commit()

with tab3:
    st.subheader("📈 Data Insights")

    st.write("### Class Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x=y, ax=ax3)
    st.pyplot(fig3)

    st.write("### Feature Correlation")
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.heatmap(X.corr(), cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    st.write("### Feature Importance")
    fig5, ax5 = plt.subplots()
    ax5.barh(X.columns, model.feature_importances_)
    st.pyplot(fig5)

    st.write("### Prediction Trends")

    rows = cursor.execute("SELECT result, probability, time FROM predictions").fetchall()

    if rows:
        df_hist = pd.DataFrame(rows, columns=["Result","Probability","Time"])

        if filter_result != "All":
            df_hist = df_hist[df_hist["Result"] == filter_result]

        st.dataframe(df_hist)

        df_hist["Time"] = pd.to_datetime(df_hist["Time"])
        df_hist = df_hist.sort_values("Time")

        fig6, ax6 = plt.subplots()
        ax6.plot(df_hist["Time"], df_hist["Probability"])
        ax6.set_title("Prediction Confidence Over Time")
        st.pyplot(fig6)
    else:
        st.info("No prediction history yet.")