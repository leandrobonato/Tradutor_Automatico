# Metodologia

## 1. Dados

**Fonte:** [`Helsinki-NLP/opus-100`](https://huggingface.co/datasets/Helsinki-NLP/opus-100),
configuração `en-pt`. O OPUS-100 é um corpus multilíngue construído a
partir da coleção [OPUS](https://opus.nlpl.eu/) — que reúne, entre
dezenas de fontes, o **Tatoeba**, o **Europarl** e o **OpenSubtitles**
citados no briefing de negócio — amostrado para 100 pares de idiomas
com o inglês como pivô.

Splits baixados (nativos do dataset) e salvos em `data/raw/`:

| Split | Pares brutos |
|---|---|
| train | 1.000.000 |
| validation | 2.000 |
| test | 2.000 |

### Limpeza e filtragem (`src/data/preprocess.py`)

1. Normalização de espaços em branco (`clean_text`).
2. Remoção de pares vazios ou duplicados.
3. Filtro de tamanho: sentenças entre 1 e 200 caracteres.
4. Filtro de razão de tamanho EN/PT: descarta pares com razão
   `max(len)/min(len) > 2.5` (indício de alinhamento ruim).
5. Subamostragem aleatória (seed fixa `42`) do split de treino para a
   meta de negócio de **~100.000 pares**.

Splits de validação e teste passam pela mesma limpeza, mas **não** são
subamostrados — preservam o tamanho nativo do OPUS-100 (2.000 cada), para
que a avaliação final reflita um conjunto de teste padrão e comparável.

## 2. Tokenização

SentencePiece BPE, um modelo por idioma, vocabulário de 16.000
subpalavras, treinado apenas no split de treino processado
(`src/data/tokenizer.py`). Tokens especiais: `<pad>=0`, `<unk>=1`,
`<s>=2`, `</s>=3`.

## 3. Modelo

Resumo dos hiperparâmetros (`src/config.py`):

| Hiperparâmetro | Valor |
|---|---|
| Embedding dim | 256 |
| Hidden dim (encoder/decoder) | 256 |
| Encoder | BiLSTM, 1 camada |
| Decoder | LSTM, 1 camada + Atenção de Bahdanau |
| Dropout | 0.3 |
| Batch size | 64 |
| Otimizador | Adam, lr=1e-3 |
| Gradient clipping | norm máxima 1.0 |
| Teacher forcing | 50% |
| Beam width (inferência) | 3 |

## 4. Treino

`src/train.py` treina por época sobre o split de treino, valida no split
de validação (sem teacher forcing) e salva o checkpoint com menor
`val_loss` em `models/seq2seq_best.pt`. Histórico de perdas por época é
salvo em `reports/training_history.json`.

## 5. Avaliação

`src/evaluate.py` traduz o split de teste inteiro com **Beam Search
(k=3)** e calcula:

- **BLEU** (`sacrebleu.corpus_bleu`) — sobreposição de n-gramas com a
  referência, métrica padrão da indústria para tradução automática.
- **chrF** (`sacrebleu.corpus_chrf`) — F-score em nível de caractere,
  mais robusto que BLEU para idiomas morfologicamente ricos como o
  português (concordância verbal/nominal, conjugações).

Resultados reais em [`RESULTS.md`](RESULTS.md).

## 6. Reprodutibilidade

Pipeline completo, do zero:

```bash
pip install -r requirements.txt
python -m src.data.download      # baixa OPUS-100 en-pt para data/raw/
python -m src.data.preprocess    # limpa e gera data/processed/
python -m src.data.tokenizer     # treina os tokenizadores SentencePiece
python -m src.train --epochs 10  # treina o modelo, salva em models/
python -m src.evaluate           # avalia no split de teste (BLEU/chrF)
python app/app.py                # demo interativa (Gradio)
```

Seed fixa (`RANDOM_SEED=42` em `src/config.py`) na subamostragem dos
dados para reprodutibilidade do dataset. Como o treino roda em CPU e usa
`teacher_forcing_ratio` estocástico, pequenas variações de loss entre
execuções são esperadas mesmo com a mesma seed de dados.
