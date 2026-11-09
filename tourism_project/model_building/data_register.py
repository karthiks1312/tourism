from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_USERNAME = os.getenv("HF_REPO_ID")
REPO_ID = f"{HF_USERNAME}/tourism-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the dataset repo if it does not exist
try:
    api.repo_info(repo_id=REPO_ID, repo_type=repo_type)
    print(f"Dataset repo '{REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{REPO_ID}' not found. Creating it...")
    create_repo(repo_id=REPO_ID, repo_type=repo_type, private=False)

local_folder = "tourism_project/data"
if not os.path.exists(local_folder):
    raise SystemExit(f"Local folder not found: {local_folder}")

api.upload_folder(
    folder_path=local_folder,
    path_in_repo="data",
    repo_id=REPO_ID,
    repo_type=repo_type,
)
print(f"Upload finished: https://huggingface.co/datasets/{REPO_ID}")
