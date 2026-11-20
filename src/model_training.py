
import pandas as pd
import joblib
import json
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_models(data):

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a Class column.")

    X = data.drop("Class", axis=1)
    y = data["Class"]

    # 70% Training, 15% Validation, 15% Testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("\nDataset Split:")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save independent test dataset
    test_data = X_test.copy()
    test_data["Class"] = y_test

    test_data.to_csv(
        DATA_PROCESSED / "test_transactions.csv",
        index=False
    )

    print("\nIndependent Test Dataset Saved!")

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                ["Time", "Amount"]
            )
        ],
        remainder="passthrough"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        filename = (
            name.lower().replace(" ", "_") + ".joblib"
        )

        joblib.dump(
            pipeline,
            MODELS_DIR / filename
        )

        print(f"{name} Saved Successfully!")

    # Tune Random Forest threshold using validation data only
    rf_model = trained_models["Random Forest"]

    val_probabilities = rf_model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.01, 1.00, 0.01)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    final_val_predictions = (
        val_probabilities >= best_threshold
    ).astype(int)

    threshold_info = {
        "threshold": best_threshold,
        "validation_precision": float(
            precision_score(
                y_val,
                final_val_predictions,
                zero_division=0
            )
        ),
        "validation_recall": float(
            recall_score(
                y_val,
                final_val_predictions,
                zero_division=0
            )
        ),
        "validation_f1": float(best_f1)
    }

    with open(
        MODELS_DIR / "random_forest_threshold.json",
        "w"
    ) as file:
        json.dump(threshold_info, file, indent=4)

    print("\nBest Random Forest Threshold:")
    print(threshold_info)

    print("\nAll Models Trained and Saved!")

    return trained_models, X_test, y_test, {
        "Random Forest": best_threshold,
        "Logistic Regression": 0.50
    }
