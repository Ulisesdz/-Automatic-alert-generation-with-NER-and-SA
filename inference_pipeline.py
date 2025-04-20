"""
inference_pipeline.py

This script performs inference on image-caption pairs by:
1. Generating image captions using BLIP.
2. Combining generated captions with existing ones.
3. Running Sentiment Analysis (SA) and Named Entity Recognition (NER).
4. Saving the results (including detected entities, sentiment, and combined text) to CSV.

The pipeline loads pretrained models for BLIP (captioning), SA (BiLSTM), and NER (BiLSTM).
"""

import os
import torch
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model utilities
from SA.utils import load_model as load_sa_model, load_word2vec
from NER.utils import load_ner as load_ner_model

# Label mappings and thresholds for SA classification
idx2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC", 9: "<PAD>"}
thresholds = (0.45, 0.55)


def preprocess(text: str) -> list[str]:
    """
    Tokenizes and lowercases input text.
    
    Args:
        text (str): The input sentence or caption.
        
    Returns:
        list[str]: Tokenized and lowercased word list.
    """
    return text.strip().lower().split()


def predict_entities(text: str) -> list[str]:
    """
    Performs NER prediction on a given text.
    
    Args:
        text (str): Input text to analyze.
        
    Returns:
        list[str]: List of recognized named entities.
    """
    tokens = preprocess(text)
    indices = [word2idx.get(tok, pad_idx) for tok in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        lengths = torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(device)
        outputs = ner_model(input_tensor, lengths)[0]
        predicted_tags = outputs.argmax(dim=-1).squeeze(0).cpu().tolist()

    tags = [idx2tag[idx] for idx in predicted_tags[:len(tokens)]]
    entities = [tok for tok, tag in zip(tokens, tags) if tag != "O"]
    return entities


def predict_sentiment(text: str) -> str:
    """
    Predicts sentiment polarity for a given text.
    
    Args:
        text (str): Input sentence or caption.
        
    Returns:
        str: Predicted sentiment class: "positive", "neutral", or "negative".
    """
    tokens = preprocess(text)
    indices = [word2vec.key_to_index.get(tok, 0) for tok in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        lengths = torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(device)
        output = sa_model(input_tensor, lengths)
        prob = torch.sigmoid(output).item()

    if prob >= thresholds[1]:
        return "positive"
    elif prob <= thresholds[0]:
        return "negative"
    else:
        return "neutral"


if __name__ == "__main__":
    # === Device configuration ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Paths configuration ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    w2v_path = os.path.join(BASE_DIR, "SA", "models", "word2vec-google-news-300.kv")
    caption_csv = os.path.join(BASE_DIR, "image_captions", "captions_output.csv")
    output_folder = os.path.join(BASE_DIR, "ner_sa_output")
    output_csv = os.path.join(output_folder, "ner_sa_output.csv")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load pretrained models 
    print("Loading Word2Vec...")
    word2vec = load_word2vec(w2v_path)
    embedding_weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

    print("Loading Sentiment Analysis model...")
    sa_model = load_sa_model(
        os.path.join(BASE_DIR, "SA", "saved_models", "model_SA_BiLSTMAtt.pth"),
        embedding_weights,
        device
    )
    sa_model.eval()

    print("Loading NER model...")
    ner_model, word2idx, tag2idx, pad_idx = load_ner_model(
        os.path.join(BASE_DIR, "NER", "saved_models", "model_NER.pth"), device
    )
    ner_model.eval()

    # Load BLIP captioning model 
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).eval().to(device)

    # Load image-caption input CSV 
    image_folder = os.path.join(BASE_DIR, "image_captions", "DATA", "IMAGES")
    caption_input_path = os.path.join(BASE_DIR, "image_captions", "captions_input.csv")
    df_input = pd.read_csv(caption_input_path)

    combined_captions = []

    for _, row in df_input.iterrows():
        img_name = row["image_name"]
        original_caption = row["caption"]
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50)
        generated = processor.decode(output[0], skip_special_tokens=True)

        combined = f"{original_caption} {generated}"
        combined_captions.append((img_name, original_caption, generated, combined))

    print("Running inference on combined captions...\n")
    results = []

    for img_name, original_caption, generated, combined_caption in combined_captions:
        entities = predict_entities(combined_caption)
        sentiment = predict_sentiment(combined_caption)

        print(f"Image: {img_name}")
        print(f"Original caption: {original_caption}")
        print(f"Generated caption: {generated}")
        print(f"Combined text: {combined_caption}")
        print(f"Entities: {entities}")
        print(f"Sentiment: {sentiment}")
        print("─" * 50)

        results.append({
            "image_name": img_name,
            "original_caption": original_caption,
            "generated_caption": generated,
            "combined_text": combined_caption,
            "entities": entities,
            "sentiment": sentiment
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
