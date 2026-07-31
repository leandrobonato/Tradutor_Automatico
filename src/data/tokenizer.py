"""Treino e carregamento de tokenizadores SentencePiece (BPE) para EN e PT.

Um modelo SentencePiece separado é treinado para cada idioma (vocabulários
independentes tendem a segmentar melhor do que um vocabulário compartilhado
quando os idiomas têm alfabetos/morfologia distintos).

Observação de ambiente: o binding C++ do SentencePiece não lida bem com
caminhos contendo caracteres acentuados no Windows (ex.: "Repositórios"),
retornando `NOT_FOUND` mesmo com o arquivo existindo. Por isso o
treino/carregamento aqui sempre passa pelos dados **em memória**
(`sentence_iterator`/`model_writer`/`load_from_serialized_proto`) em vez
de passar caminhos de arquivo diretamente para a biblioteca C++; a
leitura/escrita em disco é feita só pelo Python (`open(..., "rb"/"wb")`),
que lida com Unicode normalmente.
"""
import io
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config


def _model_path(lang: str) -> Path:
    return config.TOKENIZER_DIR / f"spm_{lang}.model"


def train_tokenizer(lang: str, vocab_size: int = config.VOCAB_SIZE) -> None:
    input_file = config.DATA_PROCESSED_DIR / f"train.{lang}"
    with open(input_file, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines),
        model_writer=model_writer,
        vocab_size=vocab_size,
        model_type=config.SPM_MODEL_TYPE,
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
        character_coverage=0.9995 if lang == "pt" else 1.0,
        shuffle_input_sentence=True,
    )

    model_path = _model_path(lang)
    model_path.write_bytes(model_writer.getvalue())

    # Também salva o .vocab (texto legível) para inspeção/documentação.
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(model_writer.getvalue())
    vocab_path = model_path.with_suffix(".vocab")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")

    print(f"[tokenizer] modelo SentencePiece treinado: {model_path} (vocab={sp.get_piece_size()})")


def load_tokenizer(lang: str) -> spm.SentencePieceProcessor:
    model_path = _model_path(lang)
    model_bytes = model_path.read_bytes()
    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(model_bytes)
    return sp


def encode(sp: spm.SentencePieceProcessor, text: str, add_bos_eos: bool = True) -> list[int]:
    ids = sp.encode(text, out_type=int)
    if add_bos_eos:
        ids = [config.BOS_ID] + ids + [config.EOS_ID]
    return ids


def decode(sp: spm.SentencePieceProcessor, ids: list[int]) -> str:
    ids = [i for i in ids if i not in (config.PAD_ID, config.BOS_ID, config.EOS_ID)]
    return sp.decode(ids)


def main() -> None:
    for lang in (config.SRC_LANG, config.TGT_LANG):
        train_tokenizer(lang)


if __name__ == "__main__":
    main()
