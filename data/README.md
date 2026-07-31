# Dados

Todos os dados ficam salvos localmente dentro desta pasta (não vão para o
git — ver `.gitignore` — mas residem fisicamente no projeto, conforme
pedido). Regeneráveis a qualquer momento via:

```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.data.tokenizer
```

## `data/raw/`

Dados brutos baixados do dataset `Helsinki-NLP/opus-100` (config
`en-pt`), um dos corpora do projeto OPUS (que reúne Tatoeba, Europarl,
OpenSubtitles, entre outras fontes).

| Arquivo | Descrição | Linhas |
|---|---|---|
| `train.en` / `train.pt` | Pares de treino brutos (paralelos, linha a linha) | 1.000.000 |
| `val.en` / `val.pt` | Pares de validação brutos | 2.000 |
| `test.en` / `test.pt` | Pares de teste brutos | 2.000 |
| `.hf_cache/` | Cache local da biblioteca `datasets` (redirecionado para dentro do projeto via `HF_HOME`) | — |

## `data/processed/`

Dados limpos, filtrados e (no caso do treino) subamostrados para a meta
de negócio de ~100.000 pares (`src/data/preprocess.py`).

| Arquivo | Descrição |
|---|---|
| `train.en` / `train.pt` | Split de treino final, ~100.000 pares |
| `val.en` / `val.pt` | Split de validação limpo |
| `test.en` / `test.pt` | Split de teste limpo (usado na avaliação final) |

## `data/tokenizer/`

Modelos SentencePiece BPE treinados (`src/data/tokenizer.py`):

| Arquivo | Descrição |
|---|---|
| `spm_en.model` / `spm_en.vocab` | Tokenizador BPE do inglês, vocab ~16k |
| `spm_pt.model` / `spm_pt.vocab` | Tokenizador BPE do português, vocab ~16k |
