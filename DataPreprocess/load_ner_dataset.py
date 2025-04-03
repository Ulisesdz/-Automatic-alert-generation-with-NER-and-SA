import os
import json
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, ClassLabel

# Ruta base
base_path = "..data/conll2003"

# Splits a procesar
splits = ["train", "test", "validation"]

for split in splits:
    split_path = os.path.join(base_path, split)
    print(f"\n Procesando split: {split}")

    # Leer dataset_info.json
    info_path = os.path.join(split_path, "dataset_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Definir los features con etiquetas legibles
    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "pos_tags": Sequence(ClassLabel(names=info["features"]["pos_tags"]["feature"]["names"])),
        "chunk_tags": Sequence(ClassLabel(names=info["features"]["chunk_tags"]["feature"]["names"])),
        "ner_tags": Sequence(ClassLabel(names=info["features"]["ner_tags"]["feature"]["names"]))
    })

    # Cargar el archivo .arrow
    arrow_filename = "data-00000-of-00001.arrow"
    arrow_path = os.path.join(split_path, arrow_filename)

    dataset = Dataset.from_file(arrow_path)
    dataset = dataset.cast(features)

    # Convertir índices a etiquetas legibles
    def decode_tags(example):
        return {
            "pos_tags": [features["pos_tags"].feature.int2str(i) for i in example["pos_tags"]],
            "chunk_tags": [features["chunk_tags"].feature.int2str(i) for i in example["chunk_tags"]],
            "ner_tags": [features["ner_tags"].feature.int2str(i) for i in example["ner_tags"]],
        }

    dataset = dataset.map(decode_tags)

    # Convertir a DataFrame para mejor formato
    df = dataset.to_pandas()

    # Formatear listas como strings legibles
    for col in ["tokens", "pos_tags", "chunk_tags", "ner_tags"]:
        df[col] = df[col].apply(lambda x: str(x))

    # Guardar CSV en la misma carpeta
    csv_output_path = os.path.join(split_path, f"{split}.csv")
    df.to_csv(csv_output_path, index=False)

    print(f"CSV guardado: {csv_output_path}")
