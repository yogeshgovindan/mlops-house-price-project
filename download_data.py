import os
import pandas as pd

# dataset URL
url = (
    "https://raw.githubusercontent.com/"
    "ageron/handson-ml/master/"
    "datasets/housing/housing.csv"
)

print("Downloading housing dataset...")

# read data directly from URL
housing_df = pd.read_csv(url)

# create data folder
os.makedirs("data", exist_ok=True)

# save dataset locally
housing_df.to_csv("data/housing.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape of dataset:", housing_df.shape)
