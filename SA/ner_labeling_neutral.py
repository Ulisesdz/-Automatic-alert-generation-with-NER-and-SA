import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt

# funciones y clases propias
from datasets import Conll2003Dataset, CollateFn
from utils import load_word2vec, load_model

# -------- Configuración --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "saved_models/model_SA_BiLSTMAtt.pth")
test_csv = os.path.join(BASE_DIR, "../data/SA+NER/test/conll2003_test_SA_neutral.csv")
word2vec_path = os.path.join(BASE_DIR, "models/word2vec-google-news-300.kv")
result_path = os.path.join(BASE_DIR, "../SA/SA+NER/results_neutral.csv")
batch_size = 64
second_threshold = (0.35, 0.65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Evaluación --------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    word2vec_model = load_word2vec(word2vec_path)
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    test_df = pd.read_csv(test_csv)

    test_dataset = Conll2003Dataset(test_csv, word2vec_model)
    test_size = len(test_dataset)
    test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size],
                                   generator=torch.Generator().manual_seed(42))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))

    model = load_model(model_path, embedding_weights, device=device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for sentences, labels, lengths in test_dataloader:
            sentences = sentences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(sentences, lengths)
            probs = torch.sigmoid(outputs).squeeze()

            pred_classes = torch.where(
                probs < second_threshold[0], torch.tensor(0, device=device),
                torch.where(probs > second_threshold[1], torch.tensor(2, device=device), torch.tensor(1, device=device))
            )

            predictions.extend(pred_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_counter = Counter(true_labels)
    pred_counter = Counter(predictions)

    print("Distribución de sentimientos modelo preentrenado:")
    print(f"Negativos: {true_counter.get(0, 0)}")
    print(f"Neutros:   {true_counter.get(1, 0)}")
    print(f"Positivos: {true_counter.get(2, 0)}")

    print("\nDistribución de sentimientos predichos:")
    print(f"Negativos: {pred_counter.get(0, 0)}")
    print(f"Neutros:   {pred_counter.get(1, 0)}")
    print(f"Positivos: {pred_counter.get(2, 0)}")

    correct = sum([1 if int(p) == int(t) else 0 for p, t in zip(predictions, true_labels)])
    total = len(true_labels)
    accuracy = correct / total * 100
    print(f"Porcentaje de coincidencias: {accuracy:.2f}%")

    plt.figure(figsize=(6, 4))
    labels = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
    true_counts = [true_counter.get(i, 0) for i in range(3)]
    pred_counts = [pred_counter.get(i, 0) for i in range(3)]

    x = range(len(labels))
    plt.bar(x, true_counts, width=0.4, label='True', align='center', alpha=0.7)
    plt.bar([i + 0.4 for i in x], pred_counts, width=0.4, label='Predicted', align='center', alpha=0.7)

    plt.xticks([i + 0.2 for i in x], labels)
    plt.title(f"Class Distribution: True vs Predicted \
    \n (Neutral: {second_threshold[0]}-{second_threshold[1]}) \n Matches: {accuracy:.2f}%")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    test_df['predicted_sentiment'] = predictions
    test_df[['sentence', 'sentiment', 'predicted_sentiment']].to_csv(result_path, index=False)

    print(f"Resultados guardados en {result_path}")
