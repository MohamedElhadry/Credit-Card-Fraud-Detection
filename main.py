
import pandas as pd

from pathlib import Path

from src.data_exploration import explore_data
from src.eda_analysis import perform_eda
from src.data_cleaning import clean_data
from src.model_training import train_models
from src.model_evaluation import evaluate_models


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)


def main():

    print("\nLoading Dataset...")

    data = pd.read_csv(DATASET_PATH)

    print("Dataset Loaded:", data.shape)

    explore_data(data)

    perform_eda(data)

    cleaned_data = clean_data(data)

    models, X_test, y_test, thresholds = train_models(
        cleaned_data
    )

    results = evaluate_models(
        models,
        X_test,
        y_test,
        thresholds
    )

    print("\n" + "=" * 60)
    print("PROJECT PIPELINE COMPLETED")
    print("=" * 60)

    print(results)


if __name__ == "__main__":
    main()
