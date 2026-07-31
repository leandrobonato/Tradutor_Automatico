"""Dataset e collate function para os pares EN->PT tokenizados."""
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.data.tokenizer import encode


class TranslationDataset(Dataset):
    def __init__(self, split_name: str, sp_en, sp_pt, max_len: int = config.MAX_DECODE_LEN):
        en_path = config.DATA_PROCESSED_DIR / f"{split_name}.en"
        pt_path = config.DATA_PROCESSED_DIR / f"{split_name}.pt"
        with open(en_path, encoding="utf-8") as f_en, open(pt_path, encoding="utf-8") as f_pt:
            en_lines = [line.strip() for line in f_en]
            pt_lines = [line.strip() for line in f_pt]

        self.pairs = []
        for en, pt in zip(en_lines, pt_lines):
            en_ids = encode(sp_en, en)[:max_len]
            pt_ids = encode(sp_pt, pt)[:max_len]
            self.pairs.append((en_ids, pt_ids))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        en_ids, pt_ids = self.pairs[idx]
        return torch.tensor(en_ids, dtype=torch.long), torch.tensor(pt_ids, dtype=torch.long)


def collate_fn(batch):
    """Faz padding dinâmico das sequências de um batch, ordenadas por
    tamanho decrescente da sequência de origem (necessário para
    `pack_padded_sequence` no encoder)."""
    batch = sorted(batch, key=lambda pair: len(pair[0]), reverse=True)
    src_seqs, tgt_seqs = zip(*batch)

    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=config.PAD_ID)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=config.PAD_ID)

    return src_padded, src_lens, tgt_padded, tgt_lens
