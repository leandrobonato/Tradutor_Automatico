"""Atenção de Bahdanau (aditiva): combina o hidden state do decoder com
cada estado do encoder para calcular pesos de alinhamento e o vetor de
contexto."""
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, encoder_output_dim: int):
        super().__init__()
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_encoder = nn.Linear(encoder_output_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor):
        """decoder_hidden: (batch, hidden_dim) — último hidden do decoder.
        encoder_outputs: (batch, src_len, encoder_output_dim).
        mask: (batch, src_len) — 1 para posições reais, 0 para padding.
        Retorna: context (batch, encoder_output_dim), attn_weights (batch, src_len).
        """
        src_len = encoder_outputs.size(1)
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.W_decoder(decoder_hidden_exp) + self.W_encoder(encoder_outputs))
        scores = self.v(energy).squeeze(2)  # (batch, src_len)

        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
