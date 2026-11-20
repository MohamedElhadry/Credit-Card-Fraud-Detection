
import streamlit as st
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Project paths
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
THRESHOLD_PATH = ROOT / "models" / "random_forest_threshold.json"

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_threshold():
    with open(THRESHOLD_PATH, "r") as file:
        return json.load(file)["threshold"]


st.title("💳 Credit Card Fraud Detection")
st.write(
    "Machine Learning system for detecting "
    "fraudulent credit card transactions."
)

# Check model files
if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Model or threshold file is missing. Run main.py first.")
    st.stop()

model = load_model()
default_threshold = float(load_threshold())

st.success("Random Forest Model Loaded Successfully!")

# Sidebar
st.sidebar.header("Model Settings")

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.01,
    max_value=0.99,
    value=default_threshold,
    step=0.01
)

st.sidebar.info(
    f"Saved Validation Threshold: {default_threshold:.2f}"
)

# Upload data
st.header("Upload Transactions")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

required_columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        if data.empty:
            st.warning("The uploaded file is empty.")
            st.stop()

        if data[required_columns].isnull().any().any():
            st.error("The uploaded data contains missing values.")
            st.stop()

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10))

        X = data[required_columns].copy()

        # Predict fraud probabilities
        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results = data.copy()

        results["Fraud_Probability"] = probabilities
        results["Predicted_Class"] = predictions

        # Metrics
        total = len(results)
        fraud_count = int(predictions.sum())
        normal_count = total - fraud_count

        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Normal Transactions", f"{normal_count:,}")
        col3.metric("Fraud Transactions", f"{fraud_count:,}")

        st.write(
            f"Fraud Percentage: "
            f"{(fraud_count / total) * 100:.3f}%"
        )

        # Chart
        st.subheader("Prediction Distribution")

        chart_data = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraud"],
            "Count": [normal_count, fraud_count]
        })

        st.bar_chart(
            chart_data.set_index("Transaction Type")
        )

        # Show predictions
        st.subheader("Detailed Predictions")

        st.dataframe(
            results,
            width="stretch"
        )

        # Evaluation only when true labels exist
        if "Class" in data.columns:

            st.header("Evaluation Against Provided Labels")

            y_true = data["Class"].astype(int)

            if not y_true.isin([0, 1]).all():
                st.error("Class must contain only 0 and 1.")
                st.stop()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Accuracy",
                f"{accuracy_score(y_true, predictions):.4f}"
            )

            col2.metric(
                "Precision",
                f"{precision_score(y_true, predictions, zero_division=0):.4f}"
            )

            col3.metric(
                "Recall",
                f"{recall_score(y_true, predictions, zero_division=0):.4f}"
            )

            col4.metric(
                "F1 Score",
                f"{f1_score(y_true, predictions, zero_division=0):.4f}"
            )

            cm = confusion_matrix(y_true, predictions)

            st.subheader("Confusion Matrix")

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Normal", "Actual Fraud"],
                columns=["Predicted Normal", "Predicted Fraud"]
            )

            st.dataframe(cm_df)

            st.warning(
                "Evaluation metrics are meaningful only when "
                "Class contains the true labels for these transactions."
            )

        # Download
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Upload a CSV file to start fraud detection.")
