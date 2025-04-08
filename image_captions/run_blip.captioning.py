import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd
import matplotlib.pyplot as plt

# ==== CONFIGURACIÓN ====
# Ruta a la carpeta test (relativa a este script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(CURRENT_DIR, "..", "data", "IMAGES", "test")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "captions_output.csv")
NUM_IMAGES = 10  # Número de imágenes a procesar (puedes cambiarlo)

# ==== CARGA DEL MODELO ====
print("Cargando modelo BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# ==== PROCESAR IMÁGENES ====
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
image_files = image_files[:NUM_IMAGES]

results = []

print(f"\nGenerando captions para {len(image_files)} imágenes...\n")

for fname in image_files:
    path = os.path.join(IMAGE_FOLDER, fname)
    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    results.append({"image_name": fname, "caption": caption})

    # Mostrar imagen + caption
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption, fontsize=12)
    plt.show()

# ==== GUARDAR RESULTADOS ====
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Captions guardadas en: {OUTPUT_CSV}")
