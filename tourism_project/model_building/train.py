# for data manipulation
import pandas as pd
import os
# preprocessing and pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# model training, tuning, evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
# serialization and hugging face
import joblib
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Tourism_MLOps_CICD_experiment")

HF_USERNAME = os.getenv("HF_REPO_ID")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-model"
MODEL_FILENAME = "best_pipeline.joblib"

api = HfApi(token=os.getenv("HF_TOKEN"))


def load(name):
    return pd.read_csv(hf_hub_download(
        repo_id=DATASET_REPO_ID, filename=name, repo_type="dataset"
    ))


Xtrain, Xtest = load("Xtrain.csv"), load("Xtest.csv")
ytrain, ytest = load("ytrain.csv").values.ravel(), load("ytest.csv").values.ravel()
print("Tourism train-test data loaded successfully")

# -------------------------------
# Feature groups
# -------------------------------
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# handle class imbalance (~19% positive)
class_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=class_weight,
    eval_metric="logloss",
)

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 1.0],
    "xgbclassifier__reg_lambda": [1, 10],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------------
# Training + MLflow logging
# -------------------------------
with mlflow.start_run(run_name="Tourism_XGBoost"):

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    metrics = {}
    for split, X_, y_ in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        pred = best_model.predict(X_)
        proba = best_model.predict_proba(X_)[:, 1]
        metrics[f"{split}_accuracy"] = accuracy_score(y_, pred)
        metrics[f"{split}_precision"] = precision_score(y_, pred)
        metrics[f"{split}_recall"] = recall_score(y_, pred)
        metrics[f"{split}_f1"] = f1_score(y_, pred)
        metrics[f"{split}_roc_auc"] = roc_auc_score(y_, proba)
    mlflow.log_metrics(metrics)
    print(metrics)

    # -------------------------------
    # Save and upload model
    # -------------------------------
    joblib.dump(best_model, MODEL_FILENAME)
    mlflow.log_artifact(MODEL_FILENAME, artifact_path="model")

    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Model repo '{MODEL_REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{MODEL_REPO_ID}'...")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=MODEL_FILENAME,
        path_in_repo=MODEL_FILENAME,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
    )
    print("Model uploaded to Hugging Face successfully")
