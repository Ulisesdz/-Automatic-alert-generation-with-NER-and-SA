from torch.utils.data import Dataset
from typing import List, Tuple
import os
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence

class NERWord2VecDataset(Dataset):
    def __init__(self, csv_path: str, word2vec_model: KeyedVectors):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo {csv_path} no fue encontrado.")

        self.word2vec = word2vec_model
        df = pd.read_csv(csv_path)

        # Procesar las oraciones y las etiquetas como listas
        self.sentences = df["tokens"].apply(lambda x: x.strip("[]").replace("'", "").split()).tolist()
        self.ner_tags = df["ner_tags"].apply(lambda x: list(map(int, x.strip("[]").split()))).tolist()
        # Para NER mapea directamente a una lista de indices por que ya esta en indices

        self.tag2idx = {
            "O": 0, "B-PER": 1, "I-PER": 2,
            "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6,
            "B-MISC": 7, "I-MISC": 8,
            "<PAD>": -1
        }


    def words_to_w2v_indices(self, sentence: List[str]) -> torch.Tensor:
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
            indices = [0]  # Manejo básico por si ninguna palabra está en Word2Vec
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        """Devuelve la cantidad de oraciones en el dataset."""
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna su frase tokenizada y los ner labels tokenizados."""
        sentence = self.sentences[idx]
        tags = self.ner_tags[idx]

        sentence_indices = self.words_to_w2v_indices(sentence) # Pasamos palabras a indices con word2vec
        tag_indices = torch.tensor(tags, dtype=torch.long) # Ya vienen en indices, pasamos solo a tensor

        return sentence_indices, tag_indices


# Collate function for padding sequences
def create_collate_fn():
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sentences, tags = zip(*batch)
        
        # Usa la misma longitud máxima para ambos
        max_len = max(len(s) for s in sentences)

        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
        padded_sentences = padded_sentences[:, :max_len]  # por si acaso
        
        padded_tags = pad_sequence(tags, batch_first=True, padding_value=-1)
        padded_tags = padded_tags[:, :max_len]  # igualamos tamaño

        lengths = torch.tensor([len(sentence) for sentence in sentences], dtype=torch.long)

        return padded_sentences, padded_tags, lengths

    return collate_fn