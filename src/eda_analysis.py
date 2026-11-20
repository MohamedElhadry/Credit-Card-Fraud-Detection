
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(data):

    os.makedirs("visualizations", exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Class Distribution
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=data,
        x="Class"
    )

    plt.title("Normal vs Fraud Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")

    plt.xticks([0, 1], ["Normal", "Fraud"])

    plt.tight_layout()
    plt.savefig("visualizations/class_distribution.png")
    plt.close()

    # 2. Class Percentage
    plt.figure(figsize=(7, 7))

    data["Class"].value_counts().plot(
        kind="pie",
        autopct="%1.2f%%",
        labels=["Normal", "Fraud"]
    )

    plt.title("Transaction Class Percentage")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig("visualizations/class_percentage.png")
    plt.close()

    # 3. Transaction Amount Distribution
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=data,
        x="Amount",
        bins=50
    )

    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")

    plt.tight_layout()
    plt.savefig("visualizations/amount_distribution.png")
    plt.close()

    # 4. Amount by Class
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=data,
        x="Class",
        y="Amount"
    )

    plt.title("Transaction Amount by Class")
    plt.xticks([0, 1], ["Normal", "Fraud"])

    plt.tight_layout()
    plt.savefig("visualizations/amount_by_class.png")
    plt.close()

    # 5. Transaction Time
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=data,
        x="Time",
        hue="Class",
        bins=50,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title("Transaction Time Distribution")
    plt.tight_layout()
    plt.savefig("visualizations/transaction_time.png")
    plt.close()

    # 6. Correlation Heatmap
    plt.figure(figsize=(18, 14))

    correlation = data.corr()

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png")
    plt.close()

    print("EDA Visualizations Created Successfully!")
