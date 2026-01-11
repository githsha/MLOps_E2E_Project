import pandas as pd

df = pd.read_csv("data/raw/reviews.csv")
df["review_text"] = df["review_text"].str.lower()
df.to_csv("data/processed/clean.csv", index=False)
print("Clean data has been saved.")
