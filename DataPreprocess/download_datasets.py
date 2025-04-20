import os
from datasets import load_dataset
import kagglehub

# Ruta base del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "../raw_data")


def create_data_folder():
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)


def download_conll2003():
    target = os.path.join(RAW_DATA_PATH, "conll2003")
    if os.path.exists(target):
        print("CoNLL-2003 dataset already exists.")
        return

    print("Downloading CoNLL-2003 from HuggingFace...")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    dataset.save_to_disk(target)
    print(f"Saved CoNLL-2003 dataset to {target}")


def download_sentiment140():
    target = os.path.join(RAW_DATA_PATH, "sentiment140")
    if os.path.exists(target):
        print("Sentiment140 dataset already exists.")
        return

    print("Downloading Sentiment140 from KaggleHub...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    os.rename(path, target)
    print(f"Moved to {target}")


if __name__ == "__main__":
    create_data_folder()
    download_conll2003()
    download_sentiment140()
