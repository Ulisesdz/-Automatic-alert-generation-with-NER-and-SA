import os
import torch
from torch.utils.data import DataLoader
from datasets import NERWord2VecDataset, create_collate_fn
from LSTM import BiLSTM
from utils import calculate_accuracy_NER, calculate_accuracy_per_tag, calculate_confusion_matrix_NER, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_csv = os.path.join(BASE_DIR, "../data/NER/test/conll2003_test.csv")
    train_csv = os.path.join(BASE_DIR, "../data/NER/train/conll2003_train.csv")
    
    # Cargar dataset de entrenamiento completo solo para obtener word2idx y tag2idx
    full_train_dataset = NERWord2VecDataset(train_csv)
    test_dataset = NERWord2VecDataset(test_csv, word2idx=full_train_dataset.word2idx)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=create_collate_fn()
    )

    # Cargar modelo
    model = load_model("mismo.pth", device=device)

    # Evaluación general
    test_acc = calculate_accuracy_NER(model, test_dataloader, device=device)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # Accuracy por etiqueta
    tag_accuracy = calculate_accuracy_per_tag(model, test_dataloader, full_train_dataset.tag2idx, device)
    print("\nAccuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: full_train_dataset.tag2idx[t]):
        info = tag_accuracy[tag]
        print(f"{str(full_train_dataset.tag2idx[tag]):8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})")

    # Matriz de confusión
    calculate_confusion_matrix_NER(model, test_dataloader, full_train_dataset.tag2idx, device=device)
