"""Inferência: tradução de texto EN->PT via Beam Search.

Uso:
    python -m src.translate --text "How are you?" --beam-width 3
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data.tokenizer import decode, encode, load_tokenizer
from src.models.seq2seq import Seq2Seq


class Hypothesis:
    __slots__ = ("tokens", "score", "hidden", "cell", "finished")

    def __init__(self, tokens, score, hidden, cell, finished=False):
        self.tokens = tokens
        self.score = score
        self.hidden = hidden
        self.cell = cell
        self.finished = finished


@torch.no_grad()
def beam_search_decode(model: Seq2Seq, src: torch.Tensor, src_lens: torch.Tensor,
                        beam_width: int = config.BEAM_WIDTH, max_len: int = config.MAX_DECODE_LEN,
                        length_penalty_alpha: float = 0.7) -> list[int]:
    """Beam search para uma única frase de origem (batch=1). Retorna a
    sequência de ids (sem <s>/</s>) da melhor hipótese."""
    model.eval()
    device = src.device

    encoder_outputs, hidden, cell = model.encoder(src, src_lens)
    mask = model.make_src_mask(src)

    beams = [Hypothesis(tokens=[config.BOS_ID], score=0.0, hidden=hidden, cell=cell)]
    completed: list[Hypothesis] = []

    for _ in range(max_len):
        candidates = []
        for hyp in beams:
            if hyp.finished:
                completed.append(hyp)
                continue

            input_token = torch.tensor([hyp.tokens[-1]], dtype=torch.long, device=device)
            logits, new_hidden, new_cell, _ = model.decoder.forward_step(
                input_token, hyp.hidden, hyp.cell, encoder_outputs, mask
            )
            log_probs = torch.log_softmax(logits, dim=1).squeeze(0)  # (vocab,)

            topk_log_probs, topk_ids = log_probs.topk(beam_width)
            for log_prob, token_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                finished = token_id == config.EOS_ID
                candidates.append(Hypothesis(
                    tokens=hyp.tokens + [token_id],
                    score=hyp.score + log_prob,
                    hidden=new_hidden, cell=new_cell,
                    finished=finished,
                ))

        if not candidates:
            break

        def normalized_score(h: Hypothesis) -> float:
            length = len(h.tokens)
            penalty = ((5 + length) / 6) ** length_penalty_alpha
            return h.score / penalty

        candidates.sort(key=normalized_score, reverse=True)
        beams = candidates[:beam_width]

        if all(hyp.finished for hyp in beams):
            completed.extend(beams)
            break

    completed.extend([h for h in beams if not h.finished])
    if not completed:
        completed = beams

    def normalized_score(h: Hypothesis) -> float:
        length = len(h.tokens)
        penalty = ((5 + length) / 6) ** length_penalty_alpha
        return h.score / penalty

    best = max(completed, key=normalized_score)
    return best.tokens[1:]  # remove <s> inicial ("</s>" removido no decode())


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Seq2Seq, "SentencePieceProcessor", "SentencePieceProcessor"]:
    sp_en = load_tokenizer(config.SRC_LANG)
    sp_pt = load_tokenizer(config.TGT_LANG)

    model = Seq2Seq(src_vocab_size=sp_en.get_piece_size(), tgt_vocab_size=sp_pt.get_piece_size())
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, sp_en, sp_pt


def translate(text: str, model: Seq2Seq, sp_en, sp_pt, device: torch.device,
              beam_width: int = config.BEAM_WIDTH) -> str:
    ids = encode(sp_en, text)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(ids)], dtype=torch.long)

    output_ids = beam_search_decode(model, src, src_lens, beam_width=beam_width)
    return decode(sp_pt, output_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Traduz uma frase de inglês para português.")
    parser.add_argument("--text", type=str, required=True, help="Texto em inglês a ser traduzido.")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    parser.add_argument("--checkpoint", type=str, default=str(config.MODELS_DIR / "seq2seq_best.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp_en, sp_pt = load_model(Path(args.checkpoint), device)

    translation = translate(args.text, model, sp_en, sp_pt, device, beam_width=args.beam_width)
    print(f"EN: {args.text}")
    print(f"PT: {translation}")


if __name__ == "__main__":
    main()
