import os
import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


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
        self.targets = torch.tensor(df["target"].tolist(), dtype=torch.float)

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


class Conll2003Dataset(Dataset):
    """
    Dataset de PyTorch para el dataset CONLL2003 con Word2Vec.
    """
    def __init__(self, csv_path: str, word2vec_model: KeyedVectors):
        """
        Inicializa el dataset cargando las oraciones y las etiquetas de sentimiento desde un archivo CSV.
        
        Args:
            csv_path (str): Ruta del archivo CSV con los datos tokenizados.
            word2vec_model (KeyedVectors): Modelo Word2Vec preentrenado.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo {csv_path} no fue encontrado.")
        
        self.word2vec = word2vec_model
        df = pd.read_csv(csv_path)

        # Procesar las oraciones y las etiquetas de sentimiento
        self.sentences = df["sentence"].fillna("").apply(lambda x: x.split()).tolist()
        # Modificación para incluir "neutral" en las etiquetas
        self.sentiments = torch.tensor(df["sentiment"].apply(self.sentiment_to_idx).tolist(), dtype=torch.long)

    def sentiment_to_idx(self, sentiment: str) -> int:
        """
        Convierte la etiqueta de sentimiento en un índice.
        0: negativo, 1: positivo, 2: neutral.
        
        Args:
            sentiment (str): Etiqueta de sentimiento.
        
        Returns:
            int: Índice correspondiente al sentimiento.
        """
        if sentiment == 'positive':
            return 1
        elif sentiment == 'negative':
            return 0
        elif sentiment == 'neutral':
            return 2
        else:
            raise ValueError(f"Sentimiento desconocido: {sentiment}")

    def word2idx(self, sentence: List[str]) -> torch.Tensor:
        """
        Convierte una lista de palabras en una lista de índices de Word2Vec.
        Se ignoran las palabras que no están en el vocabulario del modelo.

        Args:
            sentence (List[str]): Lista de tokens de una oración.

        Returns:
            torch.Tensor: Tensor con los índices de las palabras en Word2Vec.
        """
        indices = [self.word2vec.key_to_index[word] for word in sentence if word in self.word2vec.key_to_index]
        if not indices:
            indices = [0]  # Agregar padding si la oración no tiene palabras en el vocabulario
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        """Devuelve la cantidad de oraciones en el dataset."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[List[str], torch.Tensor]:
        """Retorna una oración tokenizada y su etiqueta de sentimiento."""
        return self.sentences[idx], self.sentiments[idx]
    
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
    labels = torch.tensor(labels, dtype=torch.float)

    return texts_padded, labels, lengths