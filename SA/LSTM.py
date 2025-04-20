import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    A Recurrent Neural Network (RNN) model for binary text classification.

    Utiliza embeddings preentrenados, una capa LSTM bidireccional, un mecanismo
    de atención (opcional), normalización, y un capa lineal
    para clasificar textos en dos clases (0 o 1).

    Args:
        embedding_weights (torch.Tensor): Pre-trained word embeddings.
        hidden_dim (int): Dimensión del estado oculto de la LSTM.
        num_layers (int): Número de capas LSTM.
        bidirectional (bool): Si la LSTM es bidireccional.
        dropout_p (float): Dropout aplicado en la LSTM y capa oculta final.
        output_dim (int): Salida (por defecto 1 para clasificación binaria).
        use_attention (bool): Si se aplica mecanismo de atención (True por defecto).
    """

    def __init__(self, embedding_weights: torch.Tensor, hidden_dim: int, num_layers: int,
                 bidirectional: bool, dropout_p: float, output_dim: int = 1,
                 use_attention: bool = True):
        super().__init__()

        embedding_dim = embedding_weights.shape[1]
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Embedding preentrenado (congelado)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # Capa LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Normalización y capa lineal final
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.output_layer = nn.Linear(lstm_output_dim, output_dim)

    def attention_net(self, rnn_output: torch.Tensor, final_hidden: torch.Tensor) -> torch.Tensor:
        """
        Aplica atención dot-product entre la salida de la LSTM y el estado final oculto.

        Args:
            rnn_output (Tensor): Salida de la LSTM [batch, seq_len, hidden_dim * num_directions]
            final_hidden (Tensor): Último estado oculto [batch, hidden_dim * num_directions]

        Returns:
            Tensor: Context vector atencional [batch, hidden_dim * num_directions]
        """
        attn_weights = torch.bmm(rnn_output, final_hidden.unsqueeze(2)).squeeze(2)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_output = torch.bmm(rnn_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)  # [batch, hidden*2]
        return attn_output

    def forward(self, x: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, emb_dim]

        # Empaquetar secuencia
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Concatenar estados ocultos si es bidireccional
        if self.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden*2]
        else:
            final_hidden = hidden[-1]  # [batch, hidden]

        # Atención o usar estado final directamente
        context = self.attention_net(rnn_output, final_hidden) if self.use_attention else final_hidden

        # Normalización y capa final
        norm_context = self.layer_norm(context)
        output = self.output_layer(norm_context)

        return output  # Logits sin activación