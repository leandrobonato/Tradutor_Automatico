"""Baixa o corpus paralelo EN->PT e salva dentro da pasta do projeto.

Fonte: Helsinki-NLP/opus-100 (config en-pt), um corpus construído a partir
da coleção OPUS (que reúne, entre outras, Tatoeba, Europarl e
OpenSubtitles) e amostrado para pares alinhados em 100 idiomas.

O cache do HuggingFace `datasets` é redirecionado para
`data/raw/.hf_cache` (via `HF_HOME`, setado neste módulo) para que os
dados baixados fiquem dentro da pasta do projeto, em vez do cache global
do usuário.
"""
import os
from pathlib import Path

_HF_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / ".hf_cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_HF_CACHE))

import sys

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config


def download_split(split: str) -> list[dict]:
    """Baixa um split (train/validation/test) e retorna lista de pares en/pt."""
    ds = load_dataset(
        config.HF_DATASET_NAME,
        config.HF_DATASET_CONFIG,
        split=split,
    )
    pairs = []
    for row in ds:
        translation = row["translation"]
        en = translation.get(config.SRC_LANG, "").strip()
        pt = translation.get(config.TGT_LANG, "").strip()
        if en and pt:
            pairs.append({"en": en, "pt": pt})
    return pairs


def save_pairs(pairs: list[dict], split_name: str) -> None:
    """Salva pares como dois arquivos paralelos (.en / .pt) em data/raw."""
    en_path = config.DATA_RAW_DIR / f"{split_name}.en"
    pt_path = config.DATA_RAW_DIR / f"{split_name}.pt"
    with open(en_path, "w", encoding="utf-8") as f_en, open(pt_path, "w", encoding="utf-8") as f_pt:
        for pair in pairs:
            f_en.write(pair["en"].replace("\n", " ") + "\n")
            f_pt.write(pair["pt"].replace("\n", " ") + "\n")
    print(f"[download] {split_name}: {len(pairs)} pares salvos em {en_path.name}/{pt_path.name}")


def main() -> None:
    for hf_split, out_name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        print(f"[download] baixando split '{hf_split}' de {config.HF_DATASET_NAME} ({config.HF_DATASET_CONFIG})...")
        pairs = download_split(hf_split)
        save_pairs(pairs, out_name)


if __name__ == "__main__":
    main()
