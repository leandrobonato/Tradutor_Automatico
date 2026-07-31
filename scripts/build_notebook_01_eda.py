"""Gera notebooks/01_eda.ipynb (não executado ainda)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 01 · Análise Exploratória dos Dados\n\n"
    "Exploração do corpus EN→PT processado (`data/processed/`), amostrado a partir do "
    "OPUS-100 (`Helsinki-NLP/opus-100`, config `en-pt`)."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "from pathlib import Path\n\n"
    "import matplotlib.pyplot as plt\n"
    "import pandas as pd\n\n"
    "sys.path.insert(0, str(Path.cwd().parent))\n"
    "from src import config"
))

cells.append(nbf.v4.new_markdown_cell("## Carregando os splits processados"))

cells.append(nbf.v4.new_code_cell(
    "def load_split(name):\n"
    "    with open(config.DATA_PROCESSED_DIR / f'{name}.en', encoding='utf-8') as f:\n"
    "        en = [line.strip() for line in f]\n"
    "    with open(config.DATA_PROCESSED_DIR / f'{name}.pt', encoding='utf-8') as f:\n"
    "        pt = [line.strip() for line in f]\n"
    "    return pd.DataFrame({'en': en, 'pt': pt})\n\n"
    "df_train = load_split('train')\n"
    "df_val = load_split('val')\n"
    "df_test = load_split('test')\n\n"
    "print(f'train: {len(df_train):,} pares')\n"
    "print(f'val:   {len(df_val):,} pares')\n"
    "print(f'test:  {len(df_test):,} pares')"
))

cells.append(nbf.v4.new_markdown_cell("## Amostra de pares EN→PT"))
cells.append(nbf.v4.new_code_cell("df_train.sample(10, random_state=42)"))

cells.append(nbf.v4.new_markdown_cell("## Distribuição de tamanho das sentenças (caracteres)"))
cells.append(nbf.v4.new_code_cell(
    "df_train['len_en'] = df_train['en'].str.len()\n"
    "df_train['len_pt'] = df_train['pt'].str.len()\n"
    "df_train[['len_en', 'len_pt']].describe()"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(1, 2, figsize=(11, 4))\n"
    "ax[0].hist(df_train['len_en'], bins=50, color='#4C72B0')\n"
    "ax[0].set_title('Tamanho das sentenças EN (caracteres)')\n"
    "ax[0].set_xlabel('caracteres')\n"
    "ax[1].hist(df_train['len_pt'], bins=50, color='#DD8452')\n"
    "ax[1].set_title('Tamanho das sentenças PT (caracteres)')\n"
    "ax[1].set_xlabel('caracteres')\n"
    "plt.tight_layout()\n"
    "plt.savefig(config.FIGURES_DIR / 'sentence_length_distribution.png', dpi=120)\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell("## Razão de tamanho EN/PT (sanity check do filtro de pré-processamento)"))
cells.append(nbf.v4.new_code_cell(
    "ratio = df_train[['len_en', 'len_pt']].max(axis=1) / df_train[['len_en', 'len_pt']].min(axis=1)\n"
    "print('razão máxima observada:', ratio.max())\n"
    "print(f'meta de negócio: pares de treino ~100k -> usados neste projeto: {len(df_train):,}')"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Palavras mais frequentes (top 20, após lowercase)\n"
))
cells.append(nbf.v4.new_code_cell(
    "from collections import Counter\n"
    "import re\n\n"
    "def top_words(series, n=20):\n"
    "    counter = Counter()\n"
    "    for text in series:\n"
    "        counter.update(re.findall(r\"\\w+\", text.lower()))\n"
    "    return counter.most_common(n)\n\n"
    "print('EN:', top_words(df_train['en']))\n"
    "print('PT:', top_words(df_train['pt']))"
))

nb['cells'] = cells
with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('notebooks/01_eda.ipynb gerado')
