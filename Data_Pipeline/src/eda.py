import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------------------------------
# Load merged parquet
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "grid_data_processed.parquet"

df = pd.read_parquet(DATA_PATH)

print("=" * 60)
print("Dataset Shape:", df.shape)
print("=" * 60)

# --------------------------------------------------
# Basic Info
# --------------------------------------------------
print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

# --------------------------------------------------
# Time Coverage
# --------------------------------------------------
print("\nTime Range:")
print(df["datetime"].min(), "→", df["datetime"].max())

# --------------------------------------------------
# Null Analysis
# --------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False))

# --------------------------------------------------
# Basic Statistics
# --------------------------------------------------
print("\nDescriptive Statistics:")
print(df.describe())

# --------------------------------------------------
# Zone Distribution
# --------------------------------------------------
print("\nZone Counts:")
print(df["zone"].value_counts())

# --------------------------------------------------
# Correlation Matrix (Numerical Only)
# --------------------------------------------------
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Carbon Intensity Trend
# (Replace with your actual carbon column name)
# --------------------------------------------------
carbon_col = "carbon_intensity"

if carbon_col in df.columns:
    plt.figure(figsize=(14, 5))
    for zone in df["zone"].unique():
        zone_df = df[df["zone"] == zone]
        plt.plot(zone_df["datetime"], zone_df[carbon_col], label=zone, alpha=0.6)

    plt.title("Carbon Intensity Over Time")
    plt.xlabel("Datetime")
    plt.ylabel("Carbon Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Distribution of Carbon Intensity
# --------------------------------------------------
if carbon_col in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[carbon_col], bins=50, kde=True)
    plt.title("Carbon Intensity Distribution")
    plt.tight_layout()
    plt.show()

print("\nEDA Completed.")