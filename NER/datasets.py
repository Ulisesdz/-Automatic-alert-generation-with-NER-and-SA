from torch.utils.data import Dataset
from typing import List, Tuple
import os
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence

class NERWord2VecDataset(Dataset):
    def __init__(self, csv_path: str, word2idx: dict = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"El archivo {csv_path} no fue encontrado.")

        df = pd.read_csv(csv_path)
        self.sentences = df["tokens"].apply(lambda x: x.strip("[]").replace("'", "").split()).tolist()
        self.ner_tags = df["ner_tags"].apply(lambda x: list(map(int, x.strip("[]").split()))).tolist()

        self.tag2idx = {
            "O": 0, "B-PER": 1, "I-PER": 2,
            "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6,
            "B-MISC": 7, "I-MISC": 8,
            "<PAD>": -1
        }

        self.pad_idx = 0  # Reservamos el índice 0 para <PAD>

        if word2idx is None:
            self.build_vocab()
        else:
            self.word2idx = word2idx

    def build_vocab(self):
        vocab = set(word for sentence in self.sentences for word in sentence)
        self.word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # idx + 1 to reserve 0 for PAD
        self.word2idx["<PAD>"] = self.pad_idx

    def words_to_indices(self, sentence: List[str]) -> torch.Tensor:
        indices = [self.word2idx.get(word, self.pad_idx) for word in sentence]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = self.sentences[idx]
        tags = self.ner_tags[idx]
        sentence_indices = self.words_to_indices(sentence)
        tag_indices = torch.tensor(tags, dtype=torch.long)
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