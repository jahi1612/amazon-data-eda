
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup folders
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "amazon.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("AMAZON DATA - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 2. Load dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nOriginal dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 3. Convert text values into numbers
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

numeric_columns = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = clean_numeric(df[column])

# 4. Handle missing numeric values
missing_ratings = df["rating"].isna().sum()
missing_counts = df["rating_count"].isna().sum()

if df["rating"].notna().any():
    df["rating"] = df["rating"].fillna(
        df["rating"].median()
    )

df["rating_count"] = df["rating_count"].fillna(0)

print("\nMissing values after cleaning:")
print(df[["rating", "rating_count"]].isnull().sum())
print("Ratings filled using median:", missing_ratings)
print("Rating counts filled with placeholder zero:", missing_counts)

# 5. Create new analysis columns
df["price_saving"] = (
    df["actual_price"] - df["discounted_price"]
)

df["main_category"] = (
    df["category"].astype(str)
    .str.split("|", regex=False).str[0].str.strip()
)

# 6. Descriptive statistics
print("\nNumeric summary:")
print(df[numeric_columns].describe().round(2))

# 7. Key findings
print("\n========== KEY FINDINGS ==========")
print(f"Average rating: {df['rating'].mean():.2f}")
print(f"Highest rating: {df['rating'].max():.2f}")
print(
    f"Average discount: "
    f"{df['discount_percentage'].mean():.2f}%"
)
print(f"Average actual price: Rs. {df['actual_price'].mean():.2f}")
print(
    f"Average discounted price: "
    f"Rs. {df['discounted_price'].mean():.2f}"
)
print(
    "Most common category:",
    df["main_category"].mode().iloc[0]
)

# 8. Save cleaned dataset
df.to_csv(
    OUTPUT_DIR / "cleaned_amazon_data.csv",
    index=False
)

# 9. Prepare graph style
sns.set_theme(style="whitegrid")

# Graph 1: Products by category
counts = df["main_category"].value_counts()

plt.figure(figsize=(12, 7))
sns.barplot(x=counts.values, y=counts.index)
plt.title("Number of Products by Category")
plt.xlabel("Number of Products")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "01_category_analysis.png",
    dpi=300
)
plt.close()

# Graph 2: Discount distribution
plt.figure(figsize=(9, 6))
sns.histplot(
    df["discount_percentage"].dropna(),
    bins=20, kde=True
)
plt.title("Distribution of Discounts")
plt.xlabel("Discount Percentage")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "02_discount_distribution.png",
    dpi=300
)
plt.close()

# Graph 3: Rating distribution
plt.figure(figsize=(9, 6))
sns.histplot(df["rating"].dropna(), bins=20, kde=True)
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "03_rating_distribution.png",
    dpi=300
)
plt.close()

# Graph 4: Price comparison
prices = df[
    ["actual_price", "discounted_price"]
].melt(
    var_name="Price Type",
    value_name="Price"
)

plt.figure(figsize=(9, 6))
sns.boxplot(data=prices, x="Price Type", y="Price")
plt.title("Actual Price vs Discounted Price")
plt.ylabel("Price (Rs.)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_price_comparison.png",
    dpi=300
)
plt.close()

# Graph 5: Average discount by category
discounts = (
    df.groupby("main_category")["discount_percentage"]
    .mean().sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))
sns.barplot(x=discounts.values, y=discounts.index)
plt.title("Average Discount by Category")
plt.xlabel("Average Discount (%)")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "05_category_discount.png",
    dpi=300
)
plt.close()

# Graph 6: Correlation heatmap
corr_columns = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "price_saving"
]

correlation = df[corr_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "06_correlation_heatmap.png",
    dpi=300
)
plt.close()

# 10. Category summary
category_summary = df.groupby("main_category").agg(
    product_count=("product_id", "count"),
    average_rating=("rating", "mean"),
    average_discount=("discount_percentage", "mean"),
    average_actual_price=("actual_price", "mean"),
    average_discounted_price=("discounted_price", "mean")
).round(2)

category_summary = category_summary.sort_values(
    "product_count", ascending=False
)

category_summary.to_csv(
    OUTPUT_DIR / "category_summary.csv"
)

print("\n========== CATEGORY SUMMARY ==========")
print(category_summary.to_string())

# 11. Top products by rating count
top_products = df.nlargest(10, "rating_count")[
    [
        "product_name",
        "rating",
        "rating_count",
        "actual_price",
        "discounted_price"
    ]
]

top_products.to_csv(
    OUTPUT_DIR / "top_products_by_rating_count.csv",
    index=False
)

print("\n========== TOP PRODUCTS BY RATING COUNT ==========")
print(top_products.to_string(index=False))

# 12. Highest-discounted products
top_discounts = df.nlargest(
    10, "discount_percentage"
)[
    [
        "product_name",
        "discount_percentage",
        "actual_price",
        "discounted_price"
    ]
]

top_discounts.to_csv(
    OUTPUT_DIR / "top_discounted_products.csv",
    index=False
)

print("\n========== TOP DISCOUNTS ==========")
print(top_discounts.to_string(index=False))

# 13. Final result
print("\nFinal dataset shape:", df.shape)
print("Cleaned dataset and six graphs saved.")
print("EDA COMPLETED SUCCESSFULLY!")
print("Output folder:", OUTPUT_DIR)
