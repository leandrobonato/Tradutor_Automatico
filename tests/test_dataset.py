import io

import sentencepiece as spm
import torch

from src import config
from src.data.dataset import collate_fn


def _make_tokenizer(vocab_size=32):
    lines = ["hello world", "how are you today", "good morning everyone", "thank you very much"] * 10
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines),
        model_writer=model_writer,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
    )
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(model_writer.getvalue())
    return sp


def test_collate_fn_pads_and_sorts_by_src_length():
    batch = [
        (torch.tensor([2, 5, 6, 3]), torch.tensor([2, 7, 3])),
        (torch.tensor([2, 5, 3]), torch.tensor([2, 7, 8, 9, 3])),
        (torch.tensor([2, 5, 6, 7, 8, 3]), torch.tensor([2, 7, 3])),
    ]
    src_padded, src_lens, tgt_padded, tgt_lens = collate_fn(batch)

    assert src_lens.tolist() == sorted(src_lens.tolist(), reverse=True)
    assert src_padded.shape[0] == 3
    assert src_padded.shape[1] == max(len(s) for s, _ in batch)
    assert tgt_padded.shape[0] == 3
    assert tgt_padded.shape[1] == max(len(t) for _, t in batch)


def test_collate_fn_padding_value_is_pad_id():
    batch = [
        (torch.tensor([2, 5, 6, 3]), torch.tensor([2, 7, 3])),
        (torch.tensor([2, 5, 3]), torch.tensor([2, 7, 3])),
    ]
    src_padded, _, _, _ = collate_fn(batch)
    shortest_row = src_padded[1]
    assert shortest_row[-1].item() == config.PAD_ID
