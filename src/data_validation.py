import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")
print("Rows:", len(df))
print("Missing values:")
print(df.isnull().sum())
print("Class distribution:")
print(df["sentiment"].value_counts())
