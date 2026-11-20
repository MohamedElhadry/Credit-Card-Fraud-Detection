
import pandas as pd
import os


def clean_data(data):

    print("\n" + "=" * 60)
    print("DATA CLEANING AND VALIDATION")
    print("=" * 60)

    df = data.copy()

    print("\nOriginal Rows:", len(df))

    duplicate_count = df.duplicated().sum()
    print("Duplicate Rows Found:", duplicate_count)

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    print("Rows After Removing Duplicates:", len(df))

    print("\nMissing Values:", df.isnull().sum().sum())
    print("Negative Amounts:", (df["Amount"] < 0).sum())

    print("\nTarget Distribution:")
    print(df["Class"].value_counts())

    infinite_count = (
        df.select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    print("\nInfinite Values:", infinite_count)

    os.makedirs("reports", exist_ok=True)

    report = {
        "Original Rows": len(data),
        "Rows After Cleaning": len(df),
        "Removed Duplicates": duplicate_count,
        "Missing Values": df.isnull().sum().sum(),
        "Negative Amounts": (df["Amount"] < 0).sum(),
        "Fraud Cases": (df["Class"] == 1).sum(),
        "Normal Cases": (df["Class"] == 0).sum()
    }

    pd.DataFrame(
        report.items(),
        columns=["Metric", "Value"]
    ).to_csv(
        "reports/data_quality_report.csv",
        index=False
    )

    print("\nData Quality Report Saved!")
    print("\nData Cleaning Completed!")

    return df
