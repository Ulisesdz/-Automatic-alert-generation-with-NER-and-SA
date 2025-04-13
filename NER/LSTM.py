import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, tagset_size, hidden_dim, pad_idx, pretrained_embeddings, dropout_rate):
        """
        Simplified BiLSTM model for sequence labeling (no CRF layer).

        Args:
        - embedding_dim: dimensionality of the word embeddings.
        - tagset_size: number of unique output tags.
        - hidden_dim: size of the LSTM hidden state.
        - pad_idx: index of the padding token.
        - pretrained_embeddings: preloaded embedding matrix.
        - dropout_rate: dropout probability.
        """
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.pad_idx = pad_idx

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        mask = (sentences != self.pad_idx)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        tag_scores = self.hidden2tag(lstm_out)

        # Return logits for each token position
        return tag_scores

    def predict(self, sentences):
        """
        Predict tag indices from the output logits (argmax over tag dimension).
        """
        logits = self.forward(sentences)
        predictions = torch.argmax(logits, dim=-1)
        return predictions
