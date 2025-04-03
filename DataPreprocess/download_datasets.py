import os
import requests
import zipfile
from datasets import load_dataset
import kagglehub

PATH = "raw_data"

def create_data_folder():
    global PATH
    if not os.path.exists(PATH):
        os.makedirs(PATH)

def download_conll2003():
    global PATH
    target = PATH + "/conll2003"
    if os.path.exists(target):
        print("CoNLL-2003 dataset already exists.")
        return
    
    print("Downloading CoNLL-2003 from HuggingFace...")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    dataset.save_to_disk(target)
    print(f"Saved CoNLL-2003 dataset to {target}")

def download_sentiment140():
    global PATH
    target = PATH + "/sentiment140"
    if os.path.exists(target):
        print("Sentiment140 dataset already exists.")
        return
    
    print("Downloading Sentiment140 from KaggleHub...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    os.rename(path, target)  
    print(f"Moved to {target}")

def download_flickr30k():
    global PATH
    target = PATH + "/flickr30k"
    if os.path.exists(target):
        print("Flickr30k dataset already exists.")
        return
    
    print("Downloading Flickr30k dataset from Kaggle...")
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    os.rename(path, target)
    print(f"Moved to {target}")

if __name__ == "__main__":
    create_data_folder()
    download_conll2003()
    download_sentiment140()
    download_flickr30k()