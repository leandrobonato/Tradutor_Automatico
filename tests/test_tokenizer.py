import io

import sentencepiece as spm

from src import config
from src.data.tokenizer import decode, encode


def _train_tiny_tokenizer() -> spm.SentencePieceProcessor:
    lines = ["hello world", "how are you", "this is a test sentence", "good morning"] * 20
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines),
        model_writer=model_writer,
        vocab_size=64,
        model_type="bpe",
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
    )
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(model_writer.getvalue())
    return sp


def test_encode_adds_bos_eos():
    sp = _train_tiny_tokenizer()
    ids = encode(sp, "hello world")
    assert ids[0] == config.BOS_ID
    assert ids[-1] == config.EOS_ID
    assert len(ids) > 2


def test_decode_removes_special_tokens():
    sp = _train_tiny_tokenizer()
    ids = encode(sp, "good morning")
    text = decode(sp, ids)
    assert "<s>" not in text
    assert "</s>" not in text
    assert len(text) > 0


def test_encode_decode_roundtrip_is_close():
    sp = _train_tiny_tokenizer()
    original = "how are you"
    ids = encode(sp, original)
    reconstructed = decode(sp, ids)
    assert reconstructed.strip().lower() == original.strip().lower()
