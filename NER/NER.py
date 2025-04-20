import torch
from torch.utils.data import DataLoader, random_split

# funciones y clases propias
from LSTM import BiLSTM
from utils import train_torch_model, calculate_class_weights_sklearn, evaluate
from datasets import NERWord2VecDataset, create_collate_fn

# Hiperparámetros
batch_size: int = 32
epochs: int = 50                                                        
print_every: int = 5
patience: int = 2
learning_rate: float = 0.001
hidden_dim: int = 256
num_layers: int = 2
dropout_p: float = 0.5
bidirectional: bool = True
embedding_dim = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar datasets
    train_csv = "../data/NER/train/conll2003_train.csv"
    test_csv = "../data/NER/test/conll2003_test.csv"

    # Carga solo para obtener vocabulario y etiquetas, no embeddings
    full_train_dataset = NERWord2VecDataset(train_csv)
    vocab_size = len(full_train_dataset.word2idx)  

    # Split 80/20
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, 
                                              [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))

    test_dataset = NERWord2VecDataset(test_csv, word2idx=full_train_dataset.word2idx)  # Usar mismo vocabulario

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=create_collate_fn())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=create_collate_fn())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=create_collate_fn())

    # Pesos de clase
    weights = calculate_class_weights_sklearn(full_train_dataset.tag2idx, full_train_dataset)

    # Crear modelo con capa de embedding interna
    rnn_model = BiLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        tagset_size=len(full_train_dataset.tag2idx),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_p,
        pad_idx=full_train_dataset.pad_idx
    ).to(device)

    # Función de pérdida y optimizador
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

    # Entrenamiento
    train_accuracies, val_accuracies = train_torch_model(
        rnn_model, train_dataloader, val_dataloader, criterion,
        optimizer, epochs, print_every, patience, full_train_dataset,
        device=device
    )

    # Evaluación
    evaluate(rnn_model, train_dataloader, val_dataloader, test_dataloader, device, full_train_dataset)