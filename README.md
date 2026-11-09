# 🧳 Tourism Package Prediction: End-to-End MLOps

An end-to-end machine learning project that predicts whether a customer will purchase the **Wellness Tourism Package**, so a sales team can prioritise the leads most likely to convert. It covers data versioning, preprocessing, tuned model training with experiment tracking, and automated deployment of a web app through a CI/CD pipeline.

**Model:** [karthiks969/tourism-package-model](https://huggingface.co/karthiks969/tourism-package-model) · **Dataset:** [karthiks969/tourism-dataset](https://huggingface.co/datasets/karthiks969/tourism-dataset)

## Highlights

- **Automated CI/CD** with GitHub Actions: a push to `main` runs register data → prepare data → train model → deploy app.
- **Imbalanced classification** (about 19% of customers buy): stratified split, `scale_pos_weight`, and F1-based tuning.
- **XGBoost classifier** in a scikit-learn pipeline (`StandardScaler` + `OneHotEncoder`), tuned with `GridSearchCV`.
- **MLflow** experiment tracking for every parameter combination plus final metrics.
- **Streamlit app** in Docker, deployed to a Hugging Face Space. Datasets and models are versioned on the Hugging Face Hub.

## Pipeline

```
push to main
     │
     ▼
┌──────────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ register-dataset │──▶│   data-prep   │──▶│  model-training  │──▶│  deploy-hosting  │
└──────────────────┘   └───────────────┘   └──────────────────┘   └──────────────────┘
```

## Dataset

4,128 customer records with demographics (age, gender, occupation, income), trip history, and sales-interaction details (pitch duration, follow-ups, product pitched). The target is `ProdTaken` (1 = purchased).

Cleaning: dropped the index and `CustomerID` columns and merged the inconsistent `Fe Male` label into `Female`.

## Results

Test set is a stratified 20% hold-out (`random_state=42`).

| Metric | Test |
|---|---|
| Accuracy | 0.943 |
| Precision | 0.884 |
| Recall | 0.811 |
| F1 | **0.846** |
| ROC-AUC | 0.963 |

The model fits the training set perfectly (train F1 = 1.0), so it overfits somewhat. Regularisation or cross-validated early stopping is a clear next step.

Best parameters: `n_estimators=200`, `max_depth=7`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=1.0`, `reg_lambda=1`.

## Project Structure

```
.
├── .github/workflows/pipeline.yml
└── tourism_project/
    ├── data/tourism.csv
    ├── model_building/        # data_register.py, prep.py, train.py
    ├── hosting/hosting.py     # deploys the app to a HF Space
    ├── deployment/            # Dockerfile, app.py (Streamlit), requirements.txt
    └── requirements.txt
```

## Running Locally

```bash
pip install -r tourism_project/requirements.txt

export HF_TOKEN=<hugging-face-token>
export HF_REPO_ID=<hugging-face-username>

python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
mlflow ui --port 5000 &
python tourism_project/model_building/train.py
```

## CI/CD Secrets

Add `HF_TOKEN` and `HF_REPO_ID` under **Settings → Secrets and variables → Actions**. Add the same two as secrets on the Hugging Face Space.

## Author

**Karthik S**: [Hugging Face](https://huggingface.co/karthiks969)
