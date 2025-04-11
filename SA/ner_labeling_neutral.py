import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

# funciones y clases propias
from datasets import Conll2003Dataset, CollateFn
from utils import load_word2vec, load_model

# -------- Configuración --------
model_path = "saved_models/model_SA_neutral.pth"
test_csv = "../data/SA+NER/test/conll2003_test_SA_neutral.csv"
word2vec_path = "models/word2vec-google-news-300.kv"
batch_size = 64
result_path = "../SA/SA+NER/results_neutral.csv"
second_threshold = (0.45, 0.55)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Evaluación --------
if __name__ == "__main__":
    os.makedirs(os.path.dirname("../SA/SA+NER"), exist_ok=True)
    
    # Cargar el modelo Word2Vec
    word2vec_model = load_word2vec()
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    # Leer el CSV solo para guardar resultados más tarde
    test_df = pd.read_csv(test_csv)

    # Dataset y DataLoader
    test_dataset = Conll2003Dataset(test_csv, word2vec_model)
    test_size = len(test_dataset)
    test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size],
                                   generator=torch.Generator().manual_seed(42))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))

    # Cargar el modelo entrenado
    model = load_model(model_path, embedding_weights, device=device)
    model.eval()

    predictions = []
    true_labels = []

    # Predecir el sentimiento para cada oración
    with torch.no_grad():
        for sentences, labels, lengths in test_dataloader:
            sentences = sentences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sentences, lengths)  # [batch_size, 1]
            probs = torch.sigmoid(outputs).squeeze()  # [batch_size]

            # Asignar clase basada en rangos de probabilidad
            pred_classes = torch.where(
                probs < second_threshold[0], torch.tensor(0, device=device),
                torch.where(probs > second_threshold[1], torch.tensor(2, device=device), torch.tensor(1, device=device))
            )

            predictions.extend(pred_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calcular el porcentaje de coincidencias
    correct = sum([1 if int(p) == int(t) else 0 for p, t in zip(predictions, true_labels)])
    total = len(true_labels)
    accuracy = correct / total * 100
    print(f"Porcentaje de coincidencias: {accuracy:.2f}%")

    # Guardar resultados
    test_df['predicted_sentiment'] = predictions
    test_df[['sentence', 'sentiment', 'predicted_sentiment']].to_csv(result_path, index=False)
    print(f"Resultados guardados en {result_path}")
