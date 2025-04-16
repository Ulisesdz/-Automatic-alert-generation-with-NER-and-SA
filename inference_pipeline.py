import os
import torch
import pandas as pd
import numpy as np

from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from SA.utils import load_word2vec as load_w2v_sa, load_model as load_sa_model
from NER.utils import load_word2vec as load_w2v_ner
from SA.LSTM import RNN
from NER.LSTM import BiLSTM


# configuración de las rutas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

w2v_path = "SA/models/word2vec-google-news-300.kv"
caption_csv = "image_captions/captions_output.csv"
output_folder = "ner_sa_output"
output_csv = os.path.join(output_folder, "ner_sa_output.csv")
os.makedirs(output_folder, exist_ok=True)

# loading models
print("Loading Word2Vec...")
word2vec = load_w2v_ner(w2v_path)
embedding_weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

print("Loading NER model...")
ner_model = BiLSTM(
    embedding_dim=300,
    tagset_size=10,
    hidden_dim=128,
    num_layers=3,
    dropout_rate=0.5,
    pretrained_embeddings=embedding_weights
).to(device)

ner_model.load_state_dict(torch.load("saved_models/model_NER.pth"))
ner_model.eval()

print("Loadingo SA model...")
sa_model = load_sa_model("saved_models/model_SA_neutral.pth", embedding_weights, device)
sa_model.eval()

# tags
idx2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC", 9: "<PAD>"}
thresholds = (0.45, 0.55)


def preprocess(text):
    return text.strip().lower().split()

def predict_entities(text):
    tokens = preprocess(text)
    indices = [word2vec.key_to_index.get(tok, 0) for tok in tokens]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        lengths = torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(device)
        outputs = ner_model(input_tensor, lengths)[0]

        predicted_tags = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()

    etiquetas = [idx2tag[idx] for idx in predicted_tags[:len(tokens)]]
    entidades = [tok for tok, tag in zip(tokens, etiquetas) if tag != "O"]
    return entidades

def predict_sentiment(text):
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

        

# === PROCESAR SOLO CAPTIONS (SIN IMÁGENES) ===

captions = [
    "Two men in uniform standing in front of a crowd.",
    "A woman in a red dress smiles at the camera.",
    "The company Apple is planning a big event in New York.",
    "A cat is sleeping on the sofa in a sunny room."
]

resultados = []

print("🔍 Procesando captions...\n")

for caption in captions:
    entidades = predict_entities(caption)
    sentimiento = predict_sentiment(caption)

    print(f"📝 Caption: {caption}")
    print(f"🔹 Entidades: {entidades}")
    print(f"🔹 Sentimiento: {sentimiento}")
    print("─" * 50)

    resultados.append({
        "caption": caption,
        "entities": entidades,
        "sentiment": sentimiento
    })

df_out = pd.DataFrame(resultados)
df_out.to_csv(output_csv, index=False)
print(f"\n Resultados guardados en: {output_csv}")


# # === PROCESAR CAPTIONS ===

# df = pd.read_csv(caption_csv)
# resultados = []

# print("🔍 Procesando captions...")

# for _, row in df.iterrows():
#     img = row["image_name"]
#     caption = row["caption"]

#     entidades = predict_entities(caption)
#     sentimiento = predict_sentiment(caption)

#     resultados.append({
#         "image_name": img,
#         "caption": caption,
#         "entities": entidades,
#         "sentiment": sentimiento
#     })

# df_out = pd.DataFrame(resultados)
# df_out.to_csv(output_csv, index=False)
# print(f"\n Resultados guardados en: {output_csv}")
