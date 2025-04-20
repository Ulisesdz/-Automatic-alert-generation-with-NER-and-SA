import os
import torch
from torch.utils.data import DataLoader

# funciones y clases propias
from NER.datasets import NERWord2VecDataset, create_collate_fn
from NER.utils import calculate_accuracy_NER, calculate_accuracy_per_tag, calculate_confusion_matrix_NER, load_ner

# -------- Configuración --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_csv = os.path.join(BASE_DIR, "../data/NER/test/conll2003_test.csv")
word2vec_path = os.path.join(BASE_DIR, "NER/models/word2vec-google-news-300.kv")
dataset_fraction = 1.0
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    # Cargar modelo
    model, word2idx, tag2idx, pad_idx = load_ner("model_NER.pth", device="cuda" if torch.cuda.is_available() else "cpu")

    # Cargar dataset
    test_dataset = NERWord2VecDataset(test_csv, word2idx=word2idx)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn()
    )

    # Evaluación general
    test_acc = calculate_accuracy_NER(model, test_dataloader, device=device)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Accuracy por etiqueta
    tag_accuracy = calculate_accuracy_per_tag(model, test_dataloader, tag2idx, device)
    print("\nAccuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: tag2idx[t]):
        info = tag_accuracy[tag]
        print(f"{str(tag2idx[tag]):8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})")

    # Matriz de confusión
    calculate_confusion_matrix_NER(model, test_dataloader, tag2idx, device=device)
