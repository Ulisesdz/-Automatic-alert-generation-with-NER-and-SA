import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, tagset_size: int,
                 hidden_dim: int, num_layers: int, dropout_rate: float, pad_idx: int = 0):
        """
        BiLSTM model for sequence labeling using trainable embedding layer (not pretrained).
        
        Args:
        - vocab_size: total number of tokens in vocabulary.
        - embedding_dim: dimensionality of the word embeddings.
        - tagset_size: number of unique output tags.
        - hidden_dim: size of the LSTM hidden state.
        - num_layers: number of LSTM layers.
        - dropout_rate: dropout probability.
        - pad_idx: index of the padding token.
        """
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = tagset_size
        self.pad_idx = pad_idx


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Because bidirectional=True
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # hidden_dim because it's bidirectional (2*hidden//2)

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, input_ids: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - input_ids: [batch_size, seq_length]
        - text_lengths: [batch_size] lengths of the sequences before padding

        Returns:
        - tag_scores: [batch_size, seq_length, tagset_size]
        """
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]

        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed_embedded)

        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # [batch_size, seq_length, hidden_dim]

        # Apply batch normalization
        lstm_out = lstm_out.permute(0, 2, 1)                # [batch, hidden_dim, seq_len]
        lstm_out = self.batch_norm(lstm_out)               # BatchNorm across features
        lstm_out = lstm_out.permute(0, 2, 1)               # [batch, seq_len, hidden_dim]

        lstm_out = self.dropout(lstm_out)
        tag_scores = self.hidden2tag(lstm_out)             # [batch_size, seq_length, tagset_size]

        return tag_scores
