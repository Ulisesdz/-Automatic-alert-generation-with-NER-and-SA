import os
import torch
import pandas as pd

# Importación de funciones y clases
from SA.utils import load_model as load_sa_model, load_word2vec
from NER.utils import load_ner as load_ner_model

# Definir las etiquetas y umbrales
idx2tag = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC", 9: "<PAD>"}
thresholds = (0.45, 0.55)
captions = [
    "Two men in uniform standing in front of a crowd.",
    "A woman in a red dress smiles at the camera.",
    "The company Apple is planning a big event in New York.",
    "A cat is sleeping on the sofa in a sunny room."
]


def preprocess(text):
    return text.strip().lower().split()

def predict_entities(text):
    # Preprocesar el texto
    tokens = preprocess(text)
    
    # Convertir tokens a índices utilizando el 'word2idx' del modelo NER
    indices = [word2idx.get(tok, pad_idx) for tok in tokens]  # Usamos pad_idx si no está en el vocabulario
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        # Longitudes de las secuencias
        lengths = torch.tensor([input_tensor.shape[1]], dtype=torch.long).to(device)
        
        # Pasar el tensor de entrada a través del modelo NER
        outputs = ner_model(input_tensor, lengths)[0]
        
        # Obtener las predicciones de las etiquetas (etiquetas de las entidades)
        predicted_tags = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()

    # Convertir las etiquetas de vuelta a las entidades
    etiquetas = [idx2tag[idx] for idx in predicted_tags[:len(tokens)]]
    
    # Filtrar las entidades que no sean "O" (no entidad)
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

if __name__ == "__main__":
    # Configuración de las rutas absolutas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas absolutas
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    w2v_path = os.path.join(BASE_DIR, "SA", "models", "word2vec-google-news-300.kv")
    caption_csv = os.path.join(BASE_DIR, "image_captions", "captions_output.csv")
    output_folder = os.path.join(BASE_DIR, "ner_sa_output")
    output_csv = os.path.join(output_folder, "ner_sa_output.csv")

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar modelos
    print("Loading Word2Vec...")
    word2vec = load_word2vec(w2v_path)
    embedding_weights = torch.tensor(word2vec.vectors, dtype=torch.float32)

    print("Loading SA model...")
    sa_model = load_sa_model(os.path.join(BASE_DIR, "SA", "saved_models", "model_SA_BiLSTMAtt.pth"), embedding_weights, device)
    sa_model.eval()

    print("Loading NER model...")
    ner_model, word2idx, tag2idx, pad_idx = load_ner_model(os.path.join(BASE_DIR, "NER", "saved_models", "model_NER.pth"), device)
    ner_model.eval()
    resultados = []

    # === NUEVO: Cargar imágenes y generar captions con BLIP ===
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    print("Cargando modelo BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval().to(device)

    # Carpeta donde están las imágenes asociadas
    image_folder = os.path.join(BASE_DIR, "image_captions", "DATA", "IMAGES")

    # Leer captions_input.csv 
    caption_input_path = os.path.join(BASE_DIR, "image_captions", "captions_input.csv")
    df_input = pd.read_csv(caption_input_path)

    combined_captions = []

    for _, row in df_input.iterrows():
        img_name = row["image_name"]
        original_caption = row["caption"]

        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            print(f" Imagen no encontrada: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_length=50)
        generated = processor.decode(output[0], skip_special_tokens=True)

        combined = f"{original_caption} {generated}"
        combined_captions.append((img_name, original_caption, generated, combined))


    print("Procesando captions...\n")

    for img_name, original_caption, generated, combined_caption in combined_captions:
        entidades = predict_entities(combined_caption)
        sentimiento = predict_sentiment(combined_caption)

        print(f"Imagen: {img_name}")
        print(f"Caption original: {original_caption}")
        print(f"Caption generado: {generated}")
        print(f"Texto combinado: {combined_caption}")
        print(f"Entidades: {entidades}")
        print(f"Sentimiento: {sentimiento}")
        print("─" * 50)

        resultados.append({
            "image_name": img_name,
            "original_caption": original_caption,
            "generated_caption": generated,
            "combined_text": combined_caption,
            "entities": entidades,
            "sentiment": sentimiento
        })

    df_out = pd.DataFrame(resultados)
    df_out.to_csv(output_csv, index=False)
    print(f"\nResultados guardados en: {output_csv}")
