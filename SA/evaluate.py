import torch
from torch.utils.data import DataLoader, random_split

# Funciones y clases personalizadas
from LSTM import RNN
from utils import calculate_accuracy_SA
from SA import load_word2vec, Sentiment140Dataset, CollateFn

# -------- Configuración --------
model_path = "saved_models/model1.pth"
test_csv = "../data/SA/test/sentiment140_test.csv"
word2vec_path = "models/word2vec-google-news-300.kv"
dataset_fraction = 0.1
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Cargar modelo y pesos --------
def load_model(model_path: str, embedding_weights, device: str = "cpu"):
    model = RNN(
        embedding_weights=embedding_weights,
        hidden_dim=128,
        num_layers=3,
        bidirectional=True,
        dropout_p=0.3,
        output_dim=1
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado desde {model_path}")
    return model

# -------- Evaluación --------
if __name__ == "__main__":
    word2vec_model = load_word2vec()
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    test_dataset_full = Sentiment140Dataset(test_csv, word2vec_model)
    test_size = int(len(test_dataset_full) * dataset_fraction)
    test_dataset, _ = random_split(test_dataset_full, [test_size, len(test_dataset_full) - test_size],
                                   generator=torch.Generator().manual_seed(42))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))

    model = load_model(model_path, embedding_weights, device=device)
    test_acc = calculate_accuracy_SA(model, test_dataloader, device=device)

    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
