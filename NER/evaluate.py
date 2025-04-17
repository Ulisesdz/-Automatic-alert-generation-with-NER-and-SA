import os
import torch
from torch.utils.data import DataLoader
from utils import calculate_accuracy_NER, load_word2vec, calculate_accuracy_per_tag, load_ner, calculate_confusion_matrix_NER
from datasets import NERWord2VecDataset, create_collate_fn
from LSTM import BiLSTM  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Configuración --------
model_path = os.path.join(BASE_DIR, "saved_models", "model_NER.pth")
test_csv = os.path.join(BASE_DIR, "../data/NER/test/conll2003_test.csv")
word2vec_path = os.path.join(BASE_DIR, "models/word2vec-google-news-300.kv")
dataset_fraction = 1.0  # Cambiar según el tamaño del conjunto de prueba que deseas usar
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Evaluación --------
if __name__ == "__main__":
    # Cargar el modelo de Word2Vec
    word2vec_model = load_word2vec(word2vec_path)
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    # Cargar el dataset NER para la evaluación
    test_dataset_full = NERWord2VecDataset(test_csv, word2vec_model)
    test_size = int(len(test_dataset_full) * dataset_fraction)
    test_dataset = torch.utils.data.Subset(test_dataset_full, range(test_size))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=create_collate_fn())

    # Cargar el modelo NER
    ner_model = load_ner(model_path, embedding_weights, device=device)

    # Evaluación del modelo
    test_acc = calculate_accuracy_NER(ner_model, test_dataloader, device=device)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Evaluación detallada por etiquetas
    full_train_dataset = NERWord2VecDataset(os.path.join(BASE_DIR, "../data/NER/train/conll2003_train.csv"), word2vec_model)
    tag_accuracy = calculate_accuracy_per_tag(ner_model, test_dataloader, full_train_dataset.tag2idx, device)

    print("\nAccuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: full_train_dataset.tag2idx[t]):
        info = tag_accuracy[tag]
        print(f"{str(full_train_dataset.tag2idx[tag]):8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})")

    # Calcular la matriz de confusión para NER
    calculate_confusion_matrix_NER(ner_model, test_dataloader, full_train_dataset.tag2idx, device=device)
