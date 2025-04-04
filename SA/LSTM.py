import torch.nn as nn
import torch

class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model implemented using PyTorch for text classification.

    This model utilizes an embedding layer with pre-trained weights, followed by an LSTM layer
    for processing sequential data, and a linear layer for classification.

    Attributes:
        embedding (nn.Embedding): Embedding layer initialized with pre-trained weights.
        rnn (nn.LSTM): LSTM (Long Short Term Memory) layer for processing sequential data.
        fc (nn.Linear): Linear layer for classification.

    Args:
        embedding_weights (torch.Tensor): Pre-trained word embeddings.
        hidden_dim (int): The number of features in the hidden state of the LSTM.
        num_layers (int): The number of layers in the LSTM.
    """

    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, num_layers: int, bidirectional: bool, dropout_p: float):
        """
        Initializes the RNN model with given embedding weights, hidden dimension, and number of layers.

        Args:
            embedding_weights (torch.Tensor): The pre-trained embedding weights to be used in the embedding layer.
            hidden_dim (int): The size of the hidden state in the LSTM layer.
            num_layers (int): The number of layers in the LSTM.
        """
        super().__init__()
        # TODO: Determine the embedding dimension from the embedding weights
        embedding_dim: int = embedding_weights.shape[1]

        # TODO: Create an embedding layer with the given pre-trained weights, use the Embedding.from_pretrained function
        self.embedding: nn.Embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # TODO: Initialize the LSTM layer
        self.rnn: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True  # Formato de entrada (batch, seq_len, embedding_dim)
        )

        # TODO: Create a linear layer for classification
        self.fc: nn.Linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the RNN model.

        Args:
            x (torch.Tensor): The input tensor containing word indices.
            text_lengths (torch.Tensor): Tensor containing the lengths of texts in the batch.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        # TODO: Embed the input text using the embedding layer
        embedded: torch.Tensor = self.embedding(x)

        # TODO: Pack the embedded text for efficient processing in the LSTM
        packed_embedded: torch.Tensor = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False)

        # TODO: Pass the packed sequence through the LSTM
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # TODO: Use the last hidden state for classification
        hidden: torch.Tensor = hidden[-1]

        # TODO: Return the final output after passing it through the linear layer
        return self.fc(hidden).squeeze(1)  # (batch_size,)
    

class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=3, 
                 num_layers=2, dropout_p=0.3, bidirectional=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Normalización antes de la LSTM
        self.input_bn = nn.BatchNorm1d(embedding_dim)

        # Primera capa LSTM
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Segunda capa LSTM con residual
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * (2 if bidirectional else 1),
            hidden_size=hidden_dim,
            num_layers=1,  # Solo una capa para evitar sobreajuste
            batch_first=True,
            bidirectional=bidirectional
        )

        # Tamaño de salida después de concatenar con residual
        lstm_output_size = hidden_dim * 4 if bidirectional else hidden_dim * 2

        # Capas totalmente conectadas con BatchNorm y ReLU
        self.fc = nn.Sequential(
            nn.BatchNorm1d(lstm_output_size),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, output_dim)  # 3 clases de salida
        )

    def forward(self, x, text_lengths):
        batch_size = x.size(0)

        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]

        # Normalización de entrada
        embedded = embedded.permute(0, 2, 1)  # [batch, emb_dim, seq_len]
        norm_embedded = self.input_bn(embedded)
        norm_embedded = norm_embedded.permute(0, 2, 1)  # Restaurar dimensiones

        # Inicialización de estados ocultos
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size, self.hidden_dim, device=x.device)

        # Primera LSTM
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            norm_embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm1(packed_embedded, (h0, c0))
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Segunda LSTM con residual
        lstm_out_res, _ = self.lstm2(lstm_out)
        lstm_out = torch.cat([lstm_out, lstm_out_res], dim=-1)

        # Tomar el último paso de tiempo
        output = self.fc(lstm_out[:, -1, :])

        return output  # No aplicamos softmax porque CrossEntropyLoss lo hace automáticamente