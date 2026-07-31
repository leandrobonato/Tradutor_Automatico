"""Encoder: Embedding -> BiLSTM. Produz os estados ocultos para a atenção."""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Projeta o estado final bidirecional (2*hidden) para o tamanho
        # esperado pelo decoder (hidden_dim), usado para inicializar o LSTM do decoder.
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        """src: (batch, seq_len) -> encoder_outputs: (batch, seq_len, 2*hidden),
        hidden/cell iniciais do decoder: (1, batch, hidden)."""
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        # hidden/cell: (num_layers*2, batch, hidden) -> concatena direções fwd/bwd da última camada
        hidden_fwd_bwd = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_fwd_bwd = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden0 = torch.tanh(self.fc_hidden(hidden_fwd_bwd)).unsqueeze(0)
        cell0 = torch.tanh(self.fc_cell(cell_fwd_bwd)).unsqueeze(0)

        return encoder_outputs, hidden0, cell0
