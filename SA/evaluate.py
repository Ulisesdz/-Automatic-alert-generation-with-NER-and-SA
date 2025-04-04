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
from LSTM import RNN
from utils import calculate_accuracy_SA, train_torch_model
from SA import load_word2vec, Sentiment140Dataset



# HIPERPARÁMETROS
batch_size: int = 64
epochs: int = 25
print_every: int = 1
patience: int = 50
learning_rate: float = 0.001
hidden_dim: int = 128
num_layers: int = 3
dropout_p: float = 0.3
bidirectional: bool = True

# Configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre del modelo guardado
model_name = "rnn_sentiment140.pt"  # Asegúrate de que esta ruta es correcta
model_path = f"models/{model_name}"

# Cargar modelo guardado
print(f"Cargando modelo desde {model_path}...")
rnn_model = torch.jit.load(model_path).to(device)
rnn_model.eval()  # Poner el modelo en modo evaluación

# Cargar Word2Vec
word2vec_model = load_word2vec()  # Asegúrate de que esta función está definida

# Cargar datasets
train_csv = "../data/SA/train/sentiment140_train.csv"
test_csv = "../data/SA/test/sentiment140_test.csv"

# Cargar dataset completo y dividirlo
full_train_dataset = Sentiment140Dataset(train_csv, word2vec_model)

# Usar solo el 25% de los datos para entrenamiento
dataset_fraction = 0.25
subset_size = int(len(full_train_dataset) * dataset_fraction)
full_train_subset, _ = torch.utils.data.random_split(
    full_train_dataset, [subset_size, len(full_train_dataset) - subset_size], 
    generator=torch.Generator().manual_seed(42)
)

# Dividir en train y validación (80% - 20%)
train_size = int(0.8 * len(full_train_subset))
val_size = len(full_train_subset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_subset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Cargar dataset de test
test_dataset = Sentiment140Dataset(test_csv, word2vec_model)

# Crear DataLoaders
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))

# Evaluación del modelo en cada conjunto
train_acc = calculate_accuracy_SA(rnn_model, train_dataloader, device=device)
val_acc = calculate_accuracy_SA(rnn_model, val_dataloader, device=device)
test_acc = calculate_accuracy_SA(rnn_model, test_dataloader, device=device)

print(f"\n🔹 RNN Model - Training Accuracy: {train_acc:.4f}")
print(f"🔹 RNN Model - Validation Accuracy: {val_acc:.4f}")
print(f"🔹 RNN Model - Test Accuracy: {test_acc:.4f}")

# Graficar evolución de la accuracy (si se guardaron los valores en entrenamiento)
try:
    rnn_epochs, train_accuracies = zip(*sorted(train_acc.items()))
    _, val_accuracies = zip(*sorted(val_acc.items()))

    plt.plot(rnn_epochs, train_accuracies, label='RNN Train', linestyle='-', color='blue')
    plt.plot(rnn_epochs, val_accuracies, label='RNN Validation', linestyle='--', color='blue')
    plt.axhline(y=test_acc, label='RNN Test', linestyle='-.', color='lightblue', alpha=0.5)
    plt.suptitle('Recurrent Neural Network Accuracy Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
except NameError:
    print("No se encontraron datos de entrenamiento previos para graficar.")
