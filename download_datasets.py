import os
import requests
import zipfile
from datasets import load_dataset
import kagglehub


def create_data_folder():
    if not os.path.exists("data"):
        os.makedirs("data")

def download_conll2003():
    print("Downloading CoNLL-2003 from HuggingFace...")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    dataset.save_to_disk("data/conll2003")
    print("Saved CoNLL-2003 dataset to data/conll2003")

def download_sentiment140():
    print("Downloading Sentiment140 from KaggleHub...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print(f"Downloaded Sentiment140 to {path}")
    os.rename(path, "data/sentiment140")  
    print("Moved to data/sentiment140")

def download_flickr30k():
    print("Downloading Flickr30k dataset from Kaggle...")
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    print(f"Downloaded to: {path}")
    
    target = "data/flickr30k"
    if os.path.exists(target):
        print("Flickr30k already moved.")
        return
    
    os.rename(path, target)
    print(f"Moved to: {target}")

if __name__ == "__main__":
    create_data_folder()
    download_conll2003()
    download_sentiment140()
    download_flickr30k()