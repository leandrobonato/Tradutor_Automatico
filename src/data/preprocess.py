"""Limpeza, filtragem e subamostragem dos pares EN->PT baixados.

Lê os arquivos paralelos brutos em `data/raw/{split}.{en,pt}` e produz
`data/processed/{split}.{en,pt}` já limpos, deduplicados e filtrados por
tamanho, com o split de treino subamostrado para a meta de negócio
(~100.000 pares).

O split de treino (1M linhas) é processado em streaming linha-a-linha com
*reservoir sampling* (Algoritmo R), para nunca precisar manter o dataset
inteiro filtrado em memória — importante em ambientes com RAM limitada.
"""
import random
import re
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def is_valid_pair(en: str, pt: str) -> bool:
    if not en or not pt:
        return False
    if not (config.MIN_SENTENCE_LEN_CHARS <= len(en) <= config.MAX_SENTENCE_LEN_CHARS):
        return False
    if not (config.MIN_SENTENCE_LEN_CHARS <= len(pt) <= config.MAX_SENTENCE_LEN_CHARS):
        return False
    ratio = max(len(en), len(pt)) / max(1, min(len(en), len(pt)))
    if ratio > config.MAX_LEN_RATIO:
        return False
    return True


def iter_clean_pairs(split_name: str) -> Iterator[tuple[str, str]]:
    """Gera pares (en, pt) já limpos, válidos e deduplicados, lendo os
    arquivos linha a linha (sem carregar tudo em memória)."""
    en_path = config.DATA_RAW_DIR / f"{split_name}.en"
    pt_path = config.DATA_RAW_DIR / f"{split_name}.pt"
    seen_hashes: set[int] = set()

    with open(en_path, encoding="utf-8") as f_en, open(pt_path, encoding="utf-8") as f_pt:
        for en_raw, pt_raw in zip(f_en, f_pt):
            en, pt = clean_text(en_raw), clean_text(pt_raw)
            if not is_valid_pair(en, pt):
                continue
            h = hash((en, pt))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            yield en, pt


def reservoir_sample(pairs_iter: Iterator[tuple[str, str]], sample_size: int) -> list[tuple[str, str]]:
    """Algoritmo R de reservoir sampling: amostra uniforme de `sample_size`
    itens de um stream de tamanho desconhecido, em uma única passada e
    memória O(sample_size)."""
    reservoir: list[tuple[str, str]] = []
    total_seen = 0
    for item in pairs_iter:
        total_seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, total_seen - 1)
            if j < sample_size:
                reservoir[j] = item
    return reservoir, total_seen


def save_split(pairs: list[tuple[str, str]], split_name: str) -> None:
    en_path = config.DATA_PROCESSED_DIR / f"{split_name}.en"
    pt_path = config.DATA_PROCESSED_DIR / f"{split_name}.pt"
    with open(en_path, "w", encoding="utf-8") as f_en, open(pt_path, "w", encoding="utf-8") as f_pt:
        for en, pt in pairs:
            f_en.write(en + "\n")
            f_pt.write(pt + "\n")
    print(f"[preprocess] {split_name}: {len(pairs)} pares salvos em data/processed/")


def main(train_sample_size: int = config.TRAIN_SAMPLE_SIZE) -> None:
    random.seed(config.RANDOM_SEED)

    train_pairs, total_seen = reservoir_sample(iter_clean_pairs("train"), train_sample_size)
    print(f"[preprocess] train: {total_seen} pares válidos encontrados, "
          f"{len(train_pairs)} amostrados via reservoir sampling")
    save_split(train_pairs, "train")

    for split_name in ("val", "test"):
        pairs = list(iter_clean_pairs(split_name))
        save_split(pairs, split_name)


if __name__ == "__main__":
    main()
