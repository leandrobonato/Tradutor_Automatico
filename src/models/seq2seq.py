"""Modelo Seq2Seq completo: Encoder BiLSTM + Decoder com Atenção de Bahdanau."""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 embedding_dim: int = config.EMBEDDING_DIM, hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT, pad_id: int = config.PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = config.BOS_ID
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim,
                                num_layers=config.ENCODER_LAYERS, dropout=dropout, pad_id=pad_id)
        self.decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim,
                                encoder_output_dim=hidden_dim * 2, dropout=dropout, pad_id=pad_id)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_id).long()

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = config.TEACHER_FORCING_RATIO):
        """Treino com teacher forcing. src/tgt: (batch, seq_len).
        Retorna logits (batch, tgt_len-1, vocab_size) alinhados com tgt[:, 1:]."""
        batch_size, tgt_len = tgt.size()
        device = src.device

        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_src_mask(src)

        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)
        input_token = tgt[:, 0]  # <s>

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder.forward_step(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t - 1, :] = logits

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = tgt[:, t] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_lens: torch.Tensor, max_len: int = config.MAX_DECODE_LEN):
        self.eval()
        device = src.device
        batch_size = src.size(0)

        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_src_mask(src)

        input_token = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=device)
        result = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long, device=device)

        for t in range(max_len):
            logits, hidden, cell, _ = self.decoder.forward_step(input_token, hidden, cell, encoder_outputs, mask)
            top1 = logits.argmax(1)
            result[:, t] = top1
            input_token = top1

        return result
