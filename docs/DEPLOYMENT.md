# Deployment

## Requisitos

- Python 3.10+
- Dependências em [`requirements.txt`](../requirements.txt)
- CPU é suficiente para inferência (latência por frase na casa de
  dezenas/poucas centenas de ms com beam width=3); GPU acelera
  significativamente o treino.

## Passo a passo local

```bash
git clone <repo>
cd Tradutor_Automatico
pip install -r requirements.txt

# Pipeline de dados (baixa ~1M pares, processa e treina tokenizadores)
python -m src.data.download
python -m src.data.preprocess
python -m src.data.tokenizer

# Treino (salva o melhor checkpoint em models/seq2seq_best.pt)
python -m src.train --epochs 10

# Avaliação no conjunto de teste (BLEU/chrF)
python -m src.evaluate

# Demo interativa
python app/app.py
```

## Uso via linha de comando

```bash
python -m src.translate --text "How are you today?" --beam-width 3
```

## Demo interativa (Gradio)

`app/app.py` sobe uma interface web local (Gradio) na porta 7860 por
padrão. Pode ser publicada gratuitamente no **Hugging Face Spaces**
(bastando apontar o Space para `app/app.py` e incluir `requirements.txt`
e o checkpoint `models/seq2seq_best.pt` — ou treinar o modelo direto no
Space via um *build step*).

## Empacotamento como serviço (produção)

Para um cenário de produção com volume alto (ex.: 10.000+ páginas/dia,
conforme a meta de negócio), o caminho natural é expor
`src/translate.py::translate` atrás de uma API (FastAPI/Flask), com:

- Batch de frases por requisição (o beam search já opera por frase, mas o
  encoder pode processar lotes) para melhor uso de CPU/GPU.
- Cache de traduções para textos repetidos (comum em documentos
  corporativos com trechos padronizados).
- Fila assíncrona (Celery/RQ) para processamento em lote de documentos
  grandes, desacoplando upload de tradução.

Esse serviço não foi incluído neste repositório (fora do escopo do
protótipo de modelagem), mas a função `translate()` em
[`src/translate.py`](../src/translate.py) já está isolada da camada de
interface (CLI/Gradio), pronta para ser chamada de um endpoint HTTP.

## Monitoramento de qualidade em produção

- Reavaliar BLEU/chrF periodicamente em uma amostra de traduções
  revisadas por humanos (ver [`RESULTS.md`](RESULTS.md) para os números
  de baseline).
- Registrar frases com `<unk>` no output (fora do vocabulário do
  SentencePiece) para identificar lacunas de domínio e retreinar o
  tokenizador/modelo com mais dados desse domínio quando necessário.
