import pandas as pd
import re
from transformers import pipeline
from pathlib import Path
import os
import warnings


# Función para extraer los tokens de una cadena
def extract_tokens(token_str):
    return re.findall(r"'(.*?)'", token_str)

# Función para procesar un dataset: cargar, generar frases, análisis de sentimiento y guardar
def process_dataset(input_path, output_path):
    print(f"Procesando: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error al leer el archivo {input_path}: {e}")
        return

    # Extraer tokens y unirlos en frases
    df["sentence"] = df["tokens"].apply(lambda x: " ".join(extract_tokens(x)))

    # Inicializar pipeline de análisis de sentimiento
    try:
        sentiment_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/sentiment-roberta-large-english-3-classes",
            return_all_scores=False
        )
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Analizar en batches
    sentences = df["sentence"].tolist()
    sentiments = []
    scores = []

    for i in range(0, len(sentences), 100):
        batch = sentences[i:i+100]
        results = sentiment_analyzer(batch)
        sentiments.extend([r["label"] for r in results])
        scores.extend([r["score"] for r in results])

    # Añadir resultados al DataFrame
    df["sentiment"] = sentiments
    df["sentiment_score"] = scores

    # Guardar el nuevo archivo CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Guardado en: {output_path}\n")

# Paths de entrada y salida
datasets = {
    "train": {
        "input": Path("../data/NER/train/conll2003_train.csv"),
        "output": Path("../data/SA+NER/train/conll2003_train_SA_neutral.csv"),
    },
    "validation": {
        "input": Path("../data/NER/train/conll2003_validation.csv"),
        "output": Path("../data/SA+NER/train/conll2003_validation_SA_neutral.csv"),
    },
    "test": {
        "input": Path("../data/NER/test/conll2003_test.csv"),
        "output": Path("../data/SA+NER/test/conll2003_test_SA_neutral.csv"),
    },
}

# Procesar cada dataset
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    result_path = "../data/SA+NER"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    for name, paths in datasets.items():
        process_dataset(paths["input"], paths["output"])
