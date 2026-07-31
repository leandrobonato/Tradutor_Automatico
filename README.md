# 🌐 Tradutor Automático (Inglês → Português)

Modelo de tradução automática neural (NMT) do zero: **Seq2Seq com Atenção
de Bahdanau** (Encoder BiLSTM + Decoder LSTM), tokenização SentencePiece
BPE e inferência por Beam Search — treinado em ~100.000 pares de frases
reais do corpus OPUS.

## 📋 Descrição do Negócio

Uma empresa com operações internacionais precisa traduzir documentos,
emails e comunicações internas rapidamente, sem depender de tradutores
humanos para o volume diário de texto que circula entre equipes,
clientes e fornecedores em diferentes países.

## 🎯 Objetivo do Modelo

Tradução automática (**Seq2Seq**): traduzir texto do inglês para o
português, com qualidade suficiente para uso operacional interno
(compreensão de conteúdo, triagem de documentos, comunicação rápida).

## 📊 Dados

| Item | Valor |
|---|---|
| Fonte | [OPUS](https://opus.nlpl.eu/) via `Helsinki-NLP/opus-100` (agrega Tatoeba, Europarl, OpenSubtitles e outras) |
| Pares brutos baixados | 1.000.000 (treino) + 2.000 (val) + 2.000 (teste) |
| Pares de treino usados | **100.000** (amostragem aleatória, seed fixa) |
| Pares de validação/teste | 1.811 / 1.811 (após limpeza) |
| Formato | Arquivos paralelos (`.en` / `.pt`, uma frase por linha) |
| Local dos dados | `data/raw/` (brutos) e `data/processed/` (limpos), dentro do próprio projeto |

## 🏗️ Arquitetura do Modelo

```
Seq2Seq com Attention (Bahdanau)

ENCODER:
Input: sequência inglês tokenizada (SentencePiece BPE)
    ↓
Embedding(256)
    ↓
LSTM(256, bidirectional)
    ↓
encoder_outputs (todos os estados) + hidden/cell inicial do decoder

DECODER (passo a passo):
Input: token anterior + contexto (atenção)
    ↓
Embedding(256)
    ↓
Atenção de Bahdanau sobre encoder_outputs
    ↓
LSTM(256)
    ↓
Dense(vocab_size=16.000, Softmax)
    ↓
Output: próximo token em português
```

## 🛠️ Técnicas Utilizadas

- **Tokenização:** SentencePiece (BPE), vocabulário de 16.000 subpalavras por idioma
- **Arquitetura:** Seq2Seq com Atenção de Bahdanau (aditiva)
- **Teacher Forcing:** 50% durante o treinamento
- **Inferência:** Beam Search (k=3) com normalização de comprimento
- **Métricas:** BLEU Score e chrF (`sacrebleu`)

## 📈 Resultados Esperados vs. Obtidos

| Métrica | Meta do briefing | Obtido (execução real) |
|---|---|---|
| BLEU Score | > 25 | **ver [`docs/RESULTS.md`](docs/RESULTS.md)** |
| chrF | > 55 | **ver [`docs/RESULTS.md`](docs/RESULTS.md)** |
| Vocabulário | 16.000 tokens | ✅ 16.000 (EN) / 16.000 (PT) |

> As metas do briefing (BLEU > 25, chrF > 55) são referência de mercado
> para sistemas NMT bem treinados, tipicamente com dezenas de milhões de
> pares e GPU dedicada por muitas horas/dias. Este projeto treina do
> zero, em CPU, com 100k pares — ver ressalvas de hardware e o caminho
> para atingir a meta em [`docs/RESULTS.md`](docs/RESULTS.md).

## 💼 Valor de Negócio

- **Velocidade:** ordens de magnitude mais rápido que tradução humana (segundos por página vs. minutos)
- **Custo:** fração de centavos por página processada, vs. ~R$30/página de tradução humana profissional
- **Volume:** arquitetura pensada para lotes de 10.000+ páginas/dia (ver [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md))
- **Escalabilidade:** pipeline de dados/tokenização/treino totalmente reprodutível e parametrizado para reescalar de 100k para milhões de pares conforme a demanda cresce

## 🚀 Como usar

```bash
pip install -r requirements.txt

# Pipeline de dados (opcional — já vem pronto neste repo)
python -m src.data.download
python -m src.data.preprocess
python -m src.data.tokenizer

# Treino
python -m src.train --epochs 10

# Avaliação (BLEU/chrF no conjunto de teste)
python -m src.evaluate

# Tradução via linha de comando
python -m src.translate --text "How are you today?"

# Demo interativa (Gradio)
python app/app.py
```

## 📁 Estrutura do projeto

```
Tradutor_Automatico/
├── app/app.py                      # Demo interativa (Gradio)
├── data/
│   ├── raw/                        # Dados brutos baixados (OPUS-100 en-pt)
│   ├── processed/                  # Dados limpos e amostrados
│   ├── tokenizer/                  # Modelos SentencePiece (EN/PT)
│   └── README.md                   # Dicionário de dados
├── src/
│   ├── config.py                   # Caminhos e hiperparâmetros
│   ├── data/{download,preprocess,tokenizer,dataset}.py
│   ├── models/{encoder,attention,decoder,seq2seq}.py
│   ├── train.py                    # Treino com teacher forcing
│   ├── evaluate.py                 # BLEU/chrF no conjunto de teste
│   └── translate.py                # Inferência com Beam Search
├── notebooks/                       # EDA, tokenização, treino, avaliação
├── tests/                           # Testes automatizados (pytest)
├── docs/{ARCHITECTURE,METHODOLOGY,RESULTS,DEPLOYMENT}.md
├── reports/{metrics.json, training_history.json, figures/}
├── models/seq2seq_best.pt          # Checkpoint treinado
└── requirements.txt
```

## 📚 Documentação técnica

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — metodologia de dados, treino e avaliação
- [`docs/RESULTS.md`](docs/RESULTS.md) — resultados reais obtidos
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — deploy e uso em produção
