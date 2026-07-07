import pandas as pd

train = pd.read_csv("dataset/Training_CKD_dataset.csv")

print(train[["eGFR", "Target"]].head(30))
print(train.groupby("Target")["eGFR"].describe())