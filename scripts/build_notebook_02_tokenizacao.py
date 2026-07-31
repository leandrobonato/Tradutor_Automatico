"""Gera notebooks/02_tokenizacao.ipynb (não executado ainda)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 02 · Tokenização (SentencePiece BPE)\n\n"
    "Inspeção dos tokenizadores SentencePiece BPE treinados para inglês e português "
    "(`data/tokenizer/spm_en.model`, `data/tokenizer/spm_pt.model`), vocabulário de 16.000 "
    "subpalavras cada."
))

cells.append(nbf.v4.new_code_cell(
    "import sys\n"
    "from pathlib import Path\n\n"
    "sys.path.insert(0, str(Path.cwd().parent))\n"
    "from src import config\n"
    "from src.data.tokenizer import load_tokenizer, encode, decode\n\n"
    "sp_en = load_tokenizer(config.SRC_LANG)\n"
    "sp_pt = load_tokenizer(config.TGT_LANG)\n"
    "print('vocab EN:', sp_en.get_piece_size())\n"
    "print('vocab PT:', sp_pt.get_piece_size())"
))

cells.append(nbf.v4.new_markdown_cell("## Tokens especiais"))
cells.append(nbf.v4.new_code_cell(
    "for i in range(4):\n"
    "    print(i, repr(sp_en.id_to_piece(i)))"
))

cells.append(nbf.v4.new_markdown_cell("## Exemplos de segmentação BPE"))
cells.append(nbf.v4.new_code_cell(
    "examples_en = [\n"
    "    'The company needs to translate documents quickly.',\n"
    "    'Internationalization and localization are important.',\n"
    "    'How are you today?',\n"
    "]\n"
    "for text in examples_en:\n"
    "    pieces = sp_en.encode(text, out_type=str)\n"
    "    ids = encode(sp_en, text)\n"
    "    print(f'{text!r}')\n"
    "    print('  pieces:', pieces)\n"
    "    print('  ids:   ', ids)\n"
    "    print()"
))

cells.append(nbf.v4.new_markdown_cell("## Encode/decode round-trip (português, com acentuação)"))
cells.append(nbf.v4.new_code_cell(
    "examples_pt = [\n"
    "    'A empresa precisa traduzir documentos rapidamente.',\n"
    "    'Internacionalização e localização são importantes.',\n"
    "    'Como você está hoje?',\n"
    "]\n"
    "for text in examples_pt:\n"
    "    ids = encode(sp_pt, text)\n"
    "    reconstructed = decode(sp_pt, ids)\n"
    "    print(f'original:      {text}')\n"
    "    print(f'pieces:        {sp_pt.encode(text, out_type=str)}')\n"
    "    print(f'reconstruído:  {reconstructed}')\n"
    "    print()"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Estatística: nº médio de subpalavras por sentença (split de treino)"
))
cells.append(nbf.v4.new_code_cell(
    "import numpy as np\n\n"
    "with open(config.DATA_PROCESSED_DIR / 'train.en', encoding='utf-8') as f:\n"
    "    en_lines = [line.strip() for line in f][:5000]\n"
    "with open(config.DATA_PROCESSED_DIR / 'train.pt', encoding='utf-8') as f:\n"
    "    pt_lines = [line.strip() for line in f][:5000]\n\n"
    "en_token_counts = [len(sp_en.encode(t, out_type=int)) for t in en_lines]\n"
    "pt_token_counts = [len(sp_pt.encode(t, out_type=int)) for t in pt_lines]\n\n"
    "print(f'EN: média={np.mean(en_token_counts):.1f} subpalavras/sentença, máx={max(en_token_counts)}')\n"
    "print(f'PT: média={np.mean(pt_token_counts):.1f} subpalavras/sentença, máx={max(pt_token_counts)}')"
))

nb['cells'] = cells
with open('notebooks/02_tokenizacao.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('notebooks/02_tokenizacao.ipynb gerado')
