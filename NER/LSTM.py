import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, tagset_size, hidden_dim, num_layers, pretrained_embeddings, dropout_rate, pad_idx=0):
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
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Normaliza sobre las características de salida del LSTM

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.pad_idx = pad_idx

    def forward(self, sentences: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(sentences)  # [batch, seq_len, emb_dim]

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed_embedded)

        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [batch, seq_len, hidden_dim]

        # BatchNorm espera [batch_size, num_features, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)  # [batch, hidden_dim, seq_len]
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # Volver a [batch, seq_len, hidden_dim]

        # Dropout + Linear
        lstm_out = self.dropout(lstm_out)
        tag_scores = self.hidden2tag(lstm_out)  # [batch, seq_len, tagset_size]

        return tag_scores

