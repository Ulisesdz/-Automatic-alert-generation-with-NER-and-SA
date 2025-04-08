import os
import shutil
import random
import pandas as pd

# Configuración
RAW_IMG_DIR = "raw_data/flickr30k/flickr30k_images/flickr30k_images"
CAPTION_CSV = "raw_data/flickr30k/flickr30k_images/results.csv"
OUTPUT_DIR = "data/IMAGES"
SPLITS = ["train", "val", "test"]
SPLIT_RATIO = [0.8, 0.1, 0.1]
random.seed(42)

def create_split_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def split_images():
    all_images = [f for f in os.listdir(RAW_IMG_DIR) if f.endswith(".jpg")]
    random.shuffle(all_images)
    total = len(all_images)

    train_end = int(SPLIT_RATIO[0] * total)
    val_end = train_end + int(SPLIT_RATIO[1] * total)

    split_files = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }
    return split_files

def copy_images(split_files):
    for split, files in split_files.items():
        for file in files:
            src = os.path.join(RAW_IMG_DIR, file)
            dst = os.path.join(OUTPUT_DIR, split, file)
            shutil.copy(src, dst)

def split_captions(split_files):
    df = pd.read_csv(CAPTION_CSV, sep='|')
    for split, files in split_files.items():
        df_split = df[df['image_name'].isin(files)]
        df_split.to_csv(os.path.join(OUTPUT_DIR, f"{split}_captions.csv"), index=False)

def main():
    create_split_dirs()
    split_files = split_images()
    copy_images(split_files)
    split_captions(split_files)
    print("División completada y captions guardadas por split.")

if __name__ == "__main__":
    main()
