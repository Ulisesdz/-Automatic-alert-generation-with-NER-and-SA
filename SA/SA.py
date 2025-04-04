import os
import gensim.downloader as api
import torch
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from gensim.models import KeyedVectors
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# funciones y clases propias
from LSTM import RNN, ImprovedRNN
from utils import calculate_accuracy_SA, train_torch_model



# HIPERPARÁMETROS
batch_size: int = 64
epochs: int = 30
print_every: int = 1
patience: int = 50
learning_rate: float = 0.001
hidden_dim: int = 128
num_layers: int = 3
dropout_p: float = 0.3
bidirectional: bool = True
dataset_fraction: float = 0.50


def load_word2vec(local_path="models/word2vec-google-news-300.kv"):
    """
    Carga el modelo Word2Vec preentrenado desde un archivo local si existe,
    o lo descarga desde Gensim en caso contrario.
    """
    if os.path.exists(local_path):
        print("Cargando modelo Word2Vec desde archivo local...")
        return KeyedVectors.load(local_path)
    else:
        print("Descargando modelo Word2Vec...")
        model = api.load("word2vec-google-news-300")
        # Crear la carpeta "models/" si no existe
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        model.save(local_path)
        return model


class Sentiment140Dataset(Dataset):
    """
    Dataset de PyTorch para Sentiment140 con Word2Vec.
    """

    def __init__(self, csv_path: str, word2vec_model: KeyedVectors):
        """
        Inicializa el dataset cargando los tweets y etiquetas desde un archivo CSV.

        Args:
            csv_path (str): Ruta del archivo CSV con tweets tokenizados.
            word2vec_model (KeyedVectors): Modelo Word2Vec preentrenado.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo {csv_path} no fue encontrado.")

        self.word2vec = word2vec_model
        df = pd.read_csv(csv_path)

        # Evitar valores NaN en los tweets
        self.texts = df["text"].fillna("").apply(lambda x: x.split()).tolist()
        self.targets = torch.tensor(df["target"].tolist(), dtype=torch.long)

    def word2idx(self, tweet: List[str]) -> torch.Tensor:
        """
        Convierte una lista de palabras en una lista de índices de Word2Vec.
        Se ignoran las palabras que no están en el vocabulario del modelo.

        Args:
            tweet (List[str]): Lista de tokens de un tweet.

        Returns:
            torch.Tensor: Tensor con los índices de las palabras en Word2Vec.
        """
        indices = [self.word2vec.key_to_index[word] for word in tweet if word in self.word2vec.key_to_index]
        if not indices:
            indices = [0]  # Agregar padding si el tweet no tiene palabras en el vocabulario
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        """Devuelve la cantidad de tweets en el dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[List[str], torch.Tensor]:
        """Retorna un tweet tokenizado y su etiqueta."""
        return self.texts[idx], self.targets[idx]
    

class CollateFn:
    """Clase para envolver `collate_fn` y pasar el modelo Word2Vec correctamente."""
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

    def __call__(self, batch):
        return collate_fn(batch, self.word2vec_model)


def collate_fn(
          batch: List[Tuple[List[str], int]], 
          word2vec_model: KeyedVectors
          )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Función para crear lotes con padding dinámico."""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    texts, labels = zip(*batch)

    texts_idx = [
        torch.tensor([word2vec_model.key_to_index[word] for word in tweet if word in word2vec_model.key_to_index], dtype=torch.long)
        for tweet in texts
    ]

    lengths = torch.tensor([max(len(t), 1) for t in texts_idx], dtype=torch.long)
    texts_padded = pad_sequence(texts_idx, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return texts_padded, labels, lengths





if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar Word2Vec
    word2vec_model = load_word2vec()
    print("Modelo word2vec cargado")

    # Cargar dataset completo de entrenamiento
    train_csv = "../data/SA/train/sentiment140_train.csv"
    test_csv = "../data/SA/test/sentiment140_test.csv"

    full_train_dataset = Sentiment140Dataset(train_csv, word2vec_model)

    # Dividir en 80% train y 20% val
    # train_size = int(0.8 * len(full_train_dataset))
    # val_size = len(full_train_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_train_dataset, 
    #                                           [train_size, val_size], 
    #                                           generator=torch.Generator().manual_seed(42))

    # Reduccion del dataset
    # Obtener un subconjunto del dataset completo
    subset_size = int(len(full_train_dataset) * dataset_fraction)
    full_train_subset, _ = random_split(full_train_dataset, 
                                        [subset_size, len(full_train_dataset) - subset_size], 
                                        generator=torch.Generator().manual_seed(42))

    # Ahora dividimos en train y validation
    train_size = int(0.8 * len(full_train_subset))  # 80% para entrenamiento
    val_size = len(full_train_subset) - train_size  # 20% para validación

    train_dataset, val_dataset = random_split(full_train_subset, 
                                            [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(42))

    # Cargar dataset de test
    full_test_dataset = Sentiment140Dataset(test_csv, word2vec_model)

    # Reduccion del dataset
    test_size = int(len(full_test_dataset) * dataset_fraction)
    test_dataset, _ = random_split(full_test_dataset, 
                                [test_size, len(full_test_dataset) - test_size], 
                                generator=torch.Generator().manual_seed(42))

    # Crear DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=CollateFn(word2vec_model))
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=CollateFn(word2vec_model))
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 collate_fn=CollateFn(word2vec_model))
    
    print("Dataloaders creados")

    # Crear modelo RNN
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)
    rnn_model = ImprovedRNN(embedding_weights=embedding_weights, 
                    hidden_dim=hidden_dim, 
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    dropout_p=dropout_p).to(device)
    print("Modelo RNN creado")

    # Definir función de pérdida y optimizador
    criterion = torch.nn.CrossEntropyLoss() # Para multiclase
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

    # Entrenar el modelo
    train_accuracies, val_accuracies = train_torch_model(
        rnn_model, train_dataloader, val_dataloader, criterion,
        optimizer, epochs, print_every, patience, device=device
    )

    # Evaluación final
    train_acc = calculate_accuracy_SA(rnn_model, train_dataloader, device=device)
    val_acc = calculate_accuracy_SA(rnn_model, val_dataloader, device=device)
    test_acc = calculate_accuracy_SA(rnn_model, test_dataloader, device=device)

    print(f"\n🔹 RNN Model - Training Accuracy: {train_acc:.4f}")
    print(f"🔹 RNN Model - Validation Accuracy: {val_acc:.4f}")
    print(f"🔹 RNN Model - Test Accuracy: {test_acc:.4f}")

    # Graficar evolución de la accuracy
    rnn_epochs, train_accuracies = zip(*sorted(train_accuracies.items()))
    _, val_accuracies = zip(*sorted(val_accuracies.items()))

    plt.plot(rnn_epochs, train_accuracies, label='RNN Train', linestyle='-', color='blue')
    plt.plot(rnn_epochs, val_accuracies, label='RNN Validation', linestyle='--', color='blue')
    plt.axhline(y=test_acc, label='RNN Test', linestyle='-.', color='lightblue', alpha=0.5)
    plt.suptitle('Recurrent Neural Network Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()