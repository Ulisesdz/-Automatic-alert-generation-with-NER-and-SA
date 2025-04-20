import os
import torch
from torch.utils.data import DataLoader, random_split

# Funciones y clases propias
from SA.utils import calculate_accuracy_SA, load_word2vec, load_model
from SA.datasets import Sentiment140Dataset, CollateFn

# -------- Configuración --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

test_csv = os.path.join(BASE_DIR, "../data", "SA", "test", "sentiment140_test.csv")
word2vec_path = os.path.join(BASE_DIR, "models", "word2vec-google-news-300.kv")
dataset_fraction = 0.1
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Evaluación --------
if __name__ == "__main__":

    # Cargar Word2Vec
    word2vec_model = load_word2vec(word2vec_path)
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    # Preparar dataset
    test_dataset_full = Sentiment140Dataset(test_csv, word2vec_model)
    test_size = int(len(test_dataset_full) * dataset_fraction)
    test_dataset, _ = random_split(
        test_dataset_full,
        [test_size, len(test_dataset_full) - test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Cargar datos
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CollateFn(word2vec_model)
    )

    # Cargar modelo
    model = load_model("model_SA_BiLSTMAtt.pth", embedding_weights, device=device)

    # Evaluar
    test_acc = calculate_accuracy_SA(model, test_dataloader, device=device)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
