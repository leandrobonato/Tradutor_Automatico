"""Decoder: Embedding -> LSTM com atenção de Bahdanau -> camada densa (vocab)."""
import torch
import torch.nn as nn

from src.models.attention import BahdanauAttention


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 encoder_output_dim: int, dropout: float = 0.3, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(hidden_dim, encoder_output_dim)
        self.lstm = nn.LSTM(embedding_dim + encoder_output_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_output_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                      encoder_outputs: torch.Tensor, mask: torch.Tensor):
        """Um passo de decodificação.
        input_token: (batch,) — token anterior (teacher forcing ou predição própria).
        hidden/cell: (1, batch, hidden_dim) — estado do LSTM do decoder.
        encoder_outputs: (batch, src_len, encoder_output_dim).
        mask: (batch, src_len).
        Retorna: logits (batch, vocab_size), novos hidden/cell, attn_weights (batch, src_len).
        """
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (batch, 1, emb_dim)

        context, attn_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        logits = self.fc_out(torch.cat((output, context, embedded), dim=1))

        return logits, hidden, cell, attn_weights
