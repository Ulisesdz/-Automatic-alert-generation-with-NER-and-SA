import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model for binary text classification.

    Utiliza embeddings preentrenados, una capa LSTM y una capa lineal con normalización
    para clasificar textos en dos clases (0 o 1).

    Args:
        embedding_weights (torch.Tensor): Pre-trained word embeddings.
        hidden_dim (int): Dimensión del estado oculto de la LSTM.
        num_layers (int): Número de capas LSTM.
        bidirectional (bool): Si la LSTM es bidireccional.
        dropout_p (float): Dropout aplicado en la LSTM (si num_layers > 1).
        output_dim (int): Salida (por defecto 1 para clasificación binaria).
    """

    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout_p: float, output_dim: int = 1):
        super().__init__()

        embedding_dim = embedding_weights.shape[1]

        # Embedding layer con pesos preentrenados
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

        # LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Normalización antes de la capa final
        self.bn = nn.BatchNorm1d(lstm_output_dim)

        # Capa lineal final (sin activación)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]

        # Empaquetar secuencias
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pasar por LSTM
        packed_output, (hidden, _) = self.rnn(packed_embedded)

        # Obtener último estado oculto
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1]  # [batch, hidden_dim]

        # Aplicar batchnorm
        norm_hidden = self.bn(hidden)

        # Capa final (sin activación, logits)
        output = self.fc(norm_hidden)  # [batch, 1]

        return output  # Logits sin sigmoid
