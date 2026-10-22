
# Exploratory Data Analysis on Amazon Sales Dataset Using Python

## Project Overview
This project performs Exploratory Data Analysis (EDA) on an Amazon product dataset using Python. It explores product prices, discounts, ratings, rating counts, and product categories.

## Objectives
- Clean and preprocess the dataset.
- Analyze product prices and discounts.
- Explore product ratings and rating counts.
- Compare product categories.
- Visualize patterns using charts.
- Export cleaned data and analysis summaries.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- VS Code

## Dataset
Source: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

The dataset contains 1,465 product records and 16 original columns.

## Analysis Performed
- Missing-value and duplicate checks
- Price and discount analysis
- Category-wise product analysis
- Rating distribution
- Top products by rating count
- Highest-discounted products
- Correlation heatmap

## How to Run
1. Install Python.
2. Install dependencies using:
   `pip install -r requirements.txt`
3. Place the dataset at `data/amazon.csv`.
4. Run:
   `python amazon_eda.py`

## Important Note
This dataset contains product listing information. It does not provide complete order-level transaction data, so this project does not calculate actual total sales revenue.

## Author
Student Capstone Project
