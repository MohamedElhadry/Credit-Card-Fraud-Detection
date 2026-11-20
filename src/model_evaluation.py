
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_models(models, X_test, y_test, thresholds):

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    results = []

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        probabilities = model.predict_proba(X_test)[:, 1]

        threshold = thresholds.get(name, 0.50)

        predictions = (
            probabilities >= threshold
        ).astype(int)

        metrics = {
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "PR-AUC": average_precision_score(
                y_test, probabilities
            ),
            "ROC-AUC": roc_auc_score(
                y_test, probabilities
            )
        }

        results.append(metrics)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["Normal", "Fraud"],
                zero_division=0
            )
        )

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"]
        )

        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(VISUALIZATIONS_DIR / filename)
        plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON:")
    print(results_df.to_string(index=False))

    print("\nFinal Evaluation Results Saved!")

    return results_df
