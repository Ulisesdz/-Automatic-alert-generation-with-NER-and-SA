import os
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# funciones y clases propias
from LSTM import RNN
from utils import calculate_accuracy_SA, train_torch_model, load_word2vec
from datasets import Sentiment140Dataset, CollateFn

# HIPERPARÁMETROS
batch_size: int = 64
epochs: int = 100
print_every: int = 1
patience: int = 5
learning_rate: float = 0.001
hidden_dim: int = 128
num_layers: int = 3
dropout_p: float = 0.3
bidirectional: bool = True
dataset_fraction: float = 0.1
weight_decay: float = 5e-4
use_attention: bool = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Rutas absolutas --------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(BASE_DIR, "../data/SA/train/sentiment140_train.csv")
    test_csv = os.path.join(BASE_DIR, "../data/SA/test/sentiment140_test.csv")
    word2vec_path = os.path.join(BASE_DIR, "models/word2vec-google-news-300.kv")

    # Cargar embeddings preentrenados
    word2vec_model = load_word2vec(word2vec_path)
    print("Modelo word2vec cargado")

    # Dataset y splits
    full_train_dataset = Sentiment140Dataset(train_csv, word2vec_model)
    subset_size = int(len(full_train_dataset) * dataset_fraction)
    full_train_subset, _ = random_split(full_train_dataset, [subset_size, len(full_train_dataset) - subset_size], generator=torch.Generator().manual_seed(42))

    train_size = int(0.8 * len(full_train_subset))
    val_size = len(full_train_subset) - train_size
    train_dataset, val_dataset = random_split(full_train_subset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    full_test_dataset = Sentiment140Dataset(test_csv, word2vec_model)
    test_size = int(len(full_test_dataset) * dataset_fraction)
    test_dataset, _ = random_split(full_test_dataset, [test_size, len(full_test_dataset) - test_size], generator=torch.Generator().manual_seed(42))

    # DataLoaders
    collate_fn = CollateFn(word2vec_model)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Modelo
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)
    rnn_model = RNN(
        embedding_weights=embedding_weights,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout_p=dropout_p,
        output_dim=1,
        use_attention=use_attention
    ).to(device)

    # Loss, optimizer y scheduler
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # Entrenamiento
    train_accuracies, val_accuracies = train_torch_model(
        rnn_model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        epochs,
        print_every,
        patience,
        scheduler=scheduler,
        device=device
    )

    # Evaluación final
    train_acc = calculate_accuracy_SA(rnn_model, train_dataloader, device=device)
    val_acc = calculate_accuracy_SA(rnn_model, val_dataloader, device=device)
    test_acc = calculate_accuracy_SA(rnn_model, test_dataloader, device=device)

    print(f"\nRNN Model - Training Accuracy: {train_acc:.4f}")
    print(f"RNN Model - Validation Accuracy: {val_acc:.4f}")
    print(f"RNN Model - Test Accuracy: {test_acc:.4f}")

    # Visualización
    rnn_epochs, train_accuracies = zip(*sorted(train_accuracies.items()))
    _, val_accuracies = zip(*sorted(val_accuracies.items()))

    plt.plot(rnn_epochs, train_accuracies, label='RNN Train', linestyle='-', color='blue')
    plt.plot(rnn_epochs, val_accuracies, label='RNN Validation', linestyle='--', color='blue')
    plt.axhline(y=test_acc, label='RNN Test', linestyle='-.', color='lightblue', alpha=0.5)
    plt.suptitle('Recurrent Neural Network Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()
