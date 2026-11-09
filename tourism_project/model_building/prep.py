# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# for hugging face dataset upload
from huggingface_hub import HfApi

REPO_ID = f"{os.getenv('HF_REPO_ID')}/tourism-dataset"
api = HfApi(token=os.getenv("HF_TOKEN"))

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(f"hf://datasets/{REPO_ID}/data/tourism.csv")
print("Tourism dataset loaded successfully:", df.shape)

# ---------------------------
# Clean data
# ---------------------------
# index column and customer identifier carry no signal
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
# inconsistent label for the same category
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------
# Stratified train-test split (target is imbalanced, ~19% positive)
# ---------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

for file in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=REPO_ID,
        repo_type="dataset",
    )

print("Train-test splits uploaded successfully")
