from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

REPO_ID = f"{os.getenv('HF_REPO_ID')}/Tourism-Package-Prediction"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Space if it does not exist
try:
    api.repo_info(repo_id=REPO_ID, repo_type=repo_type)
    print(f"Space '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{REPO_ID}' not found. Creating new Space...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=repo_type,
        space_sdk="docker",
        private=False,
    )

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=REPO_ID,
    repo_type=repo_type,
    path_in_repo="",
)
print("Deployment files uploaded to Hugging Face Space")
