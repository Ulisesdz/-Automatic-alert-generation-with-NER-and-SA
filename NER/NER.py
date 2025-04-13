import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
from torch.nn.utils.    rnn import pad_sequence
from LSTM import BiLSTM  # Assuming your LSTM model is in LSTM.py
from utils import calculate_accuracy_NER, train_torch_model, calculate_accuracy_per_tag, calculate_class_weights, calculate_confusion_matrix_NER  # Utility functions for accuracy and training
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

# Hyperparameters
batch_size: int = 32
epochs: int = 7
print_every: int = 5
patience: int = 10
learning_rate: float = 0.001
hidden_dim: int = 16 #16, 32, 263
num_layers: int = 2 #1, 2
dropout_p: float = 0.5
bidirectional: bool = True
embedding_dim = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class NERWord2VecDataset(Dataset):
    """
    Dataset para tareas de NER usando embeddings preentrenados (por ejemplo, Word2Vec).
    """

    def __init__(self, csv_path: str, word2vec_model: KeyedVectors):
        """
        Inicializa el dataset cargando las frases y etiquetas desde un archivo CSV.

        Args:
            csv_path (str): Ruta del archivo CSV con datos en formato CoNLL (columnas 'tokens' y 'ner_tags').
            word2vec_model (KeyedVectors): Modelo Word2Vec preentrenado.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo {csv_path} no fue encontrado.")

        self.word2vec = word2vec_model
        df = pd.read_csv(csv_path)

        self.tokens = df["tokens"].apply(lambda x: x.strip("[]").replace("'", "").split()).tolist()
        self.ner_tags = df["ner_tags"].apply(lambda x: x.strip("[]").replace("'", "").split()).tolist()

        # Crear tag2idx y añadir padding
        all_tags = sorted(set(tag for tags in self.ner_tags for tag in tags))
        self.tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
        self.tag2idx["<PAD>"] = len(self.tag2idx)

    def __len__(self) -> int:
        return len(self.tokens)

    def word2idx(self, sentence: List[str]) -> torch.Tensor:
        """
        Convierte una frase en una lista de índices de Word2Vec.

        Args:
            sentence (List[str]): Lista de tokens.

        Returns:
            torch.Tensor: Tensor con los índices de Word2Vec.
        """
        indices = [self.word2vec.key_to_index[word] for word in sentence if word in self.word2vec.key_to_index]
        if not indices:
            indices = [0]  # Padding si ningún token está en el vocabulario
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.tokens[idx]
        tags = self.ner_tags[idx]

        token_indices = self.word2idx(sentence)
        tag_indices = [self.tag2idx.get(tag, self.tag2idx["<PAD>"]) for tag in tags]

        return token_indices, torch.tensor(tag_indices)


# Collate function for padding sequences
def create_collate_fn(pad_token_idx: int, pad_tag_idx: int):
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences, tags = zip(*batch)

        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=pad_token_idx)
        padded_tags = pad_sequence(tags, batch_first=True, padding_value=pad_tag_idx)

        lengths = torch.tensor([len(sentence) for sentence in sentences], dtype=torch.long)

        return padded_sentences, padded_tags, lengths

    return collate_fn


# Pad sequences to the length of the longest sentence in the batch
# Le esta diciendo de rellenar todos los sequences (sentence y label) con <PAD> en este caso hemos elegido 0 para llegar al length del secuence de mayor longitud

"""
e.g:
 padded_sentences: [["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
                ["Peter", "Blackburn", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]]

 padded_tags: [[3, 0, 7, 0, 0, 0, 7, 0, 0],
                [1, 2, 0, 0, 0, 0, 0, 0, 0]]

 Length: [9, 2]

De esta manera tenemos todos los sentences y tags de la MISMA LONGITUD
"""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_csv = "../data/NER/train/conll2003_train.csv"
    test_csv = "../data/NER/test/conll2003_test.csv"

    word2vec_model = load_word2vec()

    full_train_dataset = NERWord2VecDataset(train_csv, word2vec_model)

    # Split dataset into 80% train and 20% validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, 
                                              [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))

    # Tomamos los indices de PAD
    pad_token_idx = full_train_dataset.token2idx["<PAD>"]
    pad_tag_idx = full_train_dataset.tag2idx["<PAD>"]

    # Load test dataset
    test_dataset = NERWord2VecDataset(test_csv)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=create_collate_fn(pad_token_idx, pad_tag_idx))
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=create_collate_fn(pad_token_idx, pad_tag_idx))
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 collate_fn=create_collate_fn(pad_token_idx, pad_tag_idx))
    print('dataloader check')
    # Create the pre-trained embeddings
    embed_file = "embeddings/GoogleNews-vectors-negative300.bin"  # Ruta a tu archivo Word2Vec preentrenado
    embedding_weights = load_skipgram_embeddings(embed_file, full_train_dataset.token2idx, embedding_dim)
    print('embedding check')

    # Calculamos los weights
    weights = calculate_class_weights(full_train_dataset.tag2idx, full_train_dataset)
    #weights = [0.1, 100, 100, 100, 100, 100, 100, 100, 100, 0.0]
    #weights = torch.tensor(weights, dtype=torch.float)
    # Crear el modelo con los embeddings preentrenados
    rnn_model = BiLSTM(
            embedding_dim=embedding_dim,
            tagset_size=len(full_train_dataset.tag2idx), # o el tamaño real de tu conjunto de etiquetas
            hidden_dim=hidden_dim,
            dropout_rate=dropout_p,
            pretrained_embeddings=embedding_weights,
            pad_idx=full_train_dataset.token2idx["<PAD>"]        # el índice del token de padding
        ).to(device)                

    # Define loss function and optimizer
    # Modificar la función de pérdida para usar los pesos de clase
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=full_train_dataset.tag2idx["<PAD>"])
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

    print("training")
    # Train the model
    train_accuracies, val_accuracies = train_torch_model(
        rnn_model, train_dataloader, val_dataloader, criterion,
        optimizer, epochs, print_every, patience, device=device
    )   

    # Final evaluation on train, validation, and test datasets
    train_acc = calculate_accuracy_NER(rnn_model, train_dataloader, device=device)
    val_acc = calculate_accuracy_NER(rnn_model, val_dataloader, device=device)
    test_acc = calculate_accuracy_NER(rnn_model, test_dataloader, device=device)

    # Print results
    print(f"\n🔹 NER Model - Training Accuracy: {train_acc:.4f}")
    print(f"🔹 NER Model - Validation Accuracy: {val_acc:.4f}")
    print(f"🔹 NER Model - Test Accuracy: {test_acc:.4f}")

    tag_accuracy = calculate_accuracy_per_tag(rnn_model, test_dataloader, full_train_dataset.tag2idx, device)
    print(tag_accuracy)
    idx2tag_symbols = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC", 9: "<PAD>"}

    print("\n🔍 Accuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: full_train_dataset.tag2idx[t]):
        info = tag_accuracy[tag]
        print(f"{idx2tag_symbols[int(tag)]:8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})")

    # Después de entrenar el modelo y realizar la evaluación
    calculate_confusion_matrix_NER(rnn_model, test_dataloader, full_train_dataset.tag2idx, device=device)
    
    # # Imprimir el mapeo de índices a etiquetas
    # print("\n Mapeo de índice a etiqueta (tag2idx):")
    # for tag, idx in sorted(full_train_dataset.tag2idx.items(), key=lambda x: x[1]):
    #     print(f"{idx:2d} → {tag}")



    # Optional: Plot accuracies over epochs
    import matplotlib.pyplot as plt

    epochs_list = list(train_accuracies.keys())
    train_acc_values = list(train_accuracies.values())
    val_acc_values = list(val_accuracies.values())

    plt.plot(epochs_list, train_acc_values, label="Training Accuracy")
    plt.plot(epochs_list, val_acc_values, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("NER Model - Accuracy over Epochs")
    plt.show()
