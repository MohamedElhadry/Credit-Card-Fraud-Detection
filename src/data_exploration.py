
import pandas as pd


def explore_data(data):

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print("Dataset Shape:", data.shape)
    print("Number of Rows:", data.shape[0])
    print("Number of Columns:", data.shape[1])

    print("\n" + "=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)

    print(data.dtypes)

    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)

    missing = data.isnull().sum()

    print(missing[missing > 0])
    print("Total Missing Values:", missing.sum())

    print("\n" + "=" * 60)
    print("4. DUPLICATE ROWS")
    print("=" * 60)

    print("Duplicate Rows:", data.duplicated().sum())

    print("\n" + "=" * 60)
    print("5. CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = data["Class"].value_counts().sort_index()

    print(class_counts)

    print("\nClass Percentages:")

    percentages = data["Class"].value_counts(
        normalize=True
    ).sort_index() * 100

    print(percentages)

    print("\n" + "=" * 60)
    print("6. STATISTICAL SUMMARY")
    print("=" * 60)

    print(data.describe())

    print("\n" + "=" * 60)
    print("7. FRAUD ANALYSIS")
    print("=" * 60)

    fraud_data = data[data["Class"] == 1]
    normal_data = data[data["Class"] == 0]

    print("Normal Transactions:", len(normal_data))
    print("Fraud Transactions:", len(fraud_data))

    print(
        "Fraud Percentage:",
        round(len(fraud_data) / len(data) * 100, 4),
        "%"
    )

    print(
        "Average Normal Transaction:",
        normal_data["Amount"].mean()
    )

    print(
        "Average Fraud Transaction:",
        fraud_data["Amount"].mean()
    )

    print("\nData Exploration Completed!")
