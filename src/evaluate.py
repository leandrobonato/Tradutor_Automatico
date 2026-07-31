"""Avaliação do modelo no conjunto de teste: BLEU e chrF (via sacrebleu),
usando Beam Search para gerar as traduções.

Uso:
    python -m src.evaluate --max-examples 500
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sacrebleu
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.translate import beam_search_decode, load_model
from src.data.tokenizer import decode as sp_decode


def evaluate(checkpoint_path: Path, split: str = "test", beam_width: int = config.BEAM_WIDTH,
             max_examples: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp_en, sp_pt = load_model(checkpoint_path, device)

    en_path = config.DATA_PROCESSED_DIR / f"{split}.en"
    pt_path = config.DATA_PROCESSED_DIR / f"{split}.pt"
    with open(en_path, encoding="utf-8") as f_en, open(pt_path, encoding="utf-8") as f_pt:
        en_lines = [line.strip() for line in f_en]
        pt_lines = [line.strip() for line in f_pt]

    if max_examples:
        en_lines = en_lines[:max_examples]
        pt_lines = pt_lines[:max_examples]

    hypotheses = []
    t0 = time.time()
    for i, en in enumerate(en_lines):
        ids = [config.BOS_ID] + sp_en.encode(en, out_type=int) + [config.EOS_ID]
        src = torch.tensor([ids], dtype=torch.long, device=device)
        src_lens = torch.tensor([len(ids)], dtype=torch.long)
        output_ids = beam_search_decode(model, src, src_lens, beam_width=beam_width)
        hypotheses.append(sp_decode(sp_pt, output_ids))
        if (i + 1) % 100 == 0:
            print(f"[evaluate] {i + 1}/{len(en_lines)} frases traduzidas...")
    elapsed = time.time() - t0

    bleu = sacrebleu.corpus_bleu(hypotheses, [pt_lines])
    chrf = sacrebleu.corpus_chrf(hypotheses, [pt_lines])

    results = {
        "split": split,
        "n_examples": len(en_lines),
        "beam_width": beam_width,
        "bleu": bleu.score,
        "chrf": chrf.score,
        "seconds_total": elapsed,
        "seconds_per_sentence": elapsed / max(1, len(en_lines)),
    }

    print(f"[evaluate] BLEU={bleu.score:.2f} | chrF={chrf.score:.2f} | n={len(en_lines)} | beam={beam_width}")

    examples = [
        {"en": en, "pt_ref": ref, "pt_pred": hyp}
        for en, ref, hyp in list(zip(en_lines, pt_lines, hypotheses))[:20]
    ]
    results["sample_translations"] = examples

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o modelo no conjunto de teste (BLEU/chrF).")
    parser.add_argument("--checkpoint", type=str, default=str(config.MODELS_DIR / "seq2seq_best.pt"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--beam-width", type=int, default=config.BEAM_WIDTH)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    results = evaluate(Path(args.checkpoint), split=args.split, beam_width=args.beam_width,
                        max_examples=args.max_examples)

    out_path = config.REPORTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] métricas salvas em {out_path}")


if __name__ == "__main__":
    main()
