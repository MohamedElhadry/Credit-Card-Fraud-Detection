
# Credit Card Fraud Detection

## 1. Project Overview
This project aims to detect fraudulent credit card transactions using Machine Learning techniques.

The dataset contains credit card transactions made by European cardholders. The main objective is to distinguish between legitimate and fraudulent transactions.

## 2. Dataset Description
The dataset contains 284,807 transactions and 31 columns.

- Time: Number of seconds elapsed between each transaction and the first transaction.
- V1 to V28: Anonymized numerical features obtained through PCA.
- Amount: Transaction amount.
- Class: Target variable.
  - 0: Normal transaction.
  - 1: Fraudulent transaction.

The dataset is highly imbalanced because fraudulent transactions represent a very small percentage of all transactions.

## 3. Project Workflow
The project follows these main stages:

1. Data Exploration
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Preprocessing
4. Train, Validation, and Test Split
5. Model Training
6. Threshold Optimization
7. Model Evaluation
8. Interactive Prediction using Streamlit

## 4. Data Preprocessing
- Checked missing values.
- Checked duplicate records.
- Removed duplicate transactions.
- Scaled the Time and Amount features.
- Used stratified splitting to preserve the class distribution.

## 5. Machine Learning Models
Two classification models were trained:

### Logistic Regression
A classification algorithm used as a baseline model.

### Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

Class weighting was used to help address the class imbalance problem.

## 6. Threshold Optimization
The Random Forest classification threshold was optimized using the validation dataset.

The selected threshold was:

0.09

The threshold was selected based on the best F1-score on the validation set.

## 7. Model Evaluation
The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Precision, Recall, and F1-score are particularly important because the dataset is highly imbalanced.

### Random Forest Test Results

- Accuracy: 0.9993
- Precision: 0.7917
- Recall: 0.8028
- F1-score: 0.7972

These results were obtained using the held-out test dataset.

## 8. Interactive Application
A Streamlit application was developed to allow users to:

- Upload transaction data in CSV format.
- Generate fraud predictions.
- View normal and fraudulent transaction counts.
- View evaluation metrics when actual labels are available.
- Inspect prediction results.
- Download the prediction results as a CSV file.

## 9. Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       └── test_transactions.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── random_forest_threshold.json
│
├── reports/
│   ├── data_quality_report.csv
│   └── model_comparison.csv
│
├── src/
│   ├── data_exploration.py
│   ├── eda_analysis.py
│   ├── data_cleaning.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── visualizations/
├── notebooks/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 10. Installation and Usage

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the complete machine learning pipeline:

```bash
python main.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 11. Limitations
- The dataset is highly imbalanced.
- The dataset contains anonymized features.
- The model is evaluated on a benchmark dataset and requires further testing before real-world deployment.
- The selected classification threshold may need adjustment depending on the cost of false positives and false negatives.

## 12. Conclusion
This project demonstrates an end-to-end machine learning workflow for credit card fraud detection, including data preprocessing, model training, threshold optimization, evaluation, and interactive predictions.
