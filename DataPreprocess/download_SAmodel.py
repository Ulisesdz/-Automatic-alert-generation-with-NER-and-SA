import os
import gdown

model_dir = "SA/saved_models"
model_path = os.path.join(model_dir, "model_SA_BiLSTMAtt.pth")

# Google Drive shareable link ID
file_id = "1lFnEVypvOVKYvHXSSQCTIpVIe0W229un"

# Crear carpeta si no existe
os.makedirs(model_dir, exist_ok=True)

# Solo descarga si no existe
if not os.path.exists(model_path):
    print("Descargando modelo desde Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
    print(f"Modelo guardado en {model_path}")
else:
    print("Modelo ya descargado.")
