import pandas as pd

train = pd.read_csv("dataset/Training_CKD_dataset.csv")
test = pd.read_csv("dataset/Testing_CKD_dataset.csv")

common = pd.merge(train, test)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Common rows:", len(common))