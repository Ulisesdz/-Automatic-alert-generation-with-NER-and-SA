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