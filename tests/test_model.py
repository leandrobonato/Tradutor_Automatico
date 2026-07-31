import torch

from src import config
from src.models.seq2seq import Seq2Seq
from src.translate import beam_search_decode

SRC_VOCAB = 50
TGT_VOCAB = 60


def _make_model():
    torch.manual_seed(0)
    return Seq2Seq(src_vocab_size=SRC_VOCAB, tgt_vocab_size=TGT_VOCAB, embedding_dim=16, hidden_dim=24)


def _make_batch(batch_size=4, src_len=7, tgt_len=5):
    torch.manual_seed(1)
    src_lens = torch.tensor(sorted([torch.randint(2, src_len + 1, (1,)).item() for _ in range(batch_size)], reverse=True))
    src = torch.zeros(batch_size, src_len, dtype=torch.long)
    for i, length in enumerate(src_lens):
        src[i, :length] = torch.randint(4, SRC_VOCAB, (length.item(),))

    tgt = torch.zeros(batch_size, tgt_len, dtype=torch.long)
    tgt[:, 0] = config.BOS_ID
    tgt[:, 1:-1] = torch.randint(4, TGT_VOCAB, (batch_size, tgt_len - 2))
    tgt[:, -1] = config.EOS_ID
    return src, src_lens, tgt


def test_encoder_output_shapes():
    model = _make_model()
    src, src_lens, _ = _make_batch()
    encoder_outputs, hidden, cell = model.encoder(src, src_lens)
    batch_size, src_len = src.shape
    assert encoder_outputs.shape == (batch_size, src_len, model.encoder.hidden_dim * 2)
    assert hidden.shape == (1, batch_size, model.encoder.hidden_dim)
    assert cell.shape == (1, batch_size, model.encoder.hidden_dim)


def test_seq2seq_forward_output_shape():
    model = _make_model()
    src, src_lens, tgt = _make_batch()
    outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)
    batch_size, tgt_len = tgt.shape
    assert outputs.shape == (batch_size, tgt_len - 1, TGT_VOCAB)


def test_seq2seq_forward_teacher_forcing_extremes_run():
    model = _make_model()
    src, src_lens, tgt = _make_batch()
    out_full_tf = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
    out_no_tf = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)
    assert out_full_tf.shape == out_no_tf.shape


def test_greedy_decode_shape():
    model = _make_model()
    src, src_lens, _ = _make_batch()
    result = model.greedy_decode(src, src_lens, max_len=10)
    assert result.shape == (src.shape[0], 10)


def test_beam_search_decode_returns_valid_sequence():
    model = _make_model()
    src = torch.randint(4, SRC_VOCAB, (1, 6), dtype=torch.long)
    src_lens = torch.tensor([6])
    tokens = beam_search_decode(model, src, src_lens, beam_width=3, max_len=10)
    assert isinstance(tokens, list)
    assert len(tokens) <= 10
    assert all(isinstance(t, int) for t in tokens)
