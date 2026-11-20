import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def clean_data(data):
    print("\n" + "=" * 60)
    print("DATA CLEANING AND VALIDATION")
    print("=" * 60)

    required_columns = {"Time", "Amount", "Class"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df = data.copy()

    print("\nOriginal Rows:", len(df))
    duplicate_count = int(df.duplicated().sum())
    print("Duplicate Rows Found:", duplicate_count)

    # Remove exact duplicate rows before splitting the data.
    df = df.drop_duplicates().copy()

    print("Rows After Removing Duplicates:", len(df))
    missing_count = int(df.isnull().sum().sum())
    negative_amounts = int((df["Amount"] < 0).sum())

    print("\nMissing Values:", missing_count)
    print("Negative Amounts:", negative_amounts)

    infinite_count = int(
        df.select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )
    print("\nInfinite Values:", infinite_count)

    if missing_count or infinite_count:
        raise ValueError(
            "Dataset contains missing or infinite values. "
            "Please inspect and resolve them before training."
        )

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    print("\nTarget Distribution:")
    print(df["Class"].value_counts())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "Original Rows": len(data),
        "Rows After Cleaning": len(df),
        "Removed Duplicates": duplicate_count,
        "Missing Values": missing_count,
        "Infinite Values": infinite_count,
        "Negative Amounts": negative_amounts,
        "Fraud Cases": int((df["Class"] == 1).sum()),
        "Normal Cases": int((df["Class"] == 0).sum()),
    }

    pd.DataFrame(
        report.items(), columns=["Metric", "Value"]
    ).to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)

    print("\nData Quality Report Saved!")
    print("\nData Cleaning Completed!")

    return df
