#!/bin/bash
cd "D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico"
LOG="notebooks_01_02_progress.log"
echo "[retry-wrapper] iniciado em $(date)" > "$LOG"
for i in $(seq 1 30); do
  echo "[retry-wrapper] tentativa $i em $(date)" >> "$LOG"
  {
    python scripts/build_notebook_01_eda.py &&
    python scripts/build_notebook_02_tokenizacao.py &&
    python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb &&
    python -m nbconvert --to notebook --execute --inplace notebooks/02_tokenizacao.ipynb
  } >> "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[retry-wrapper] SUCESSO em $(date)" >> "$LOG"
    exit 0
  fi
  echo "[retry-wrapper] falhou, aguardando 25s..." >> "$LOG"
  sleep 25
done
echo "[retry-wrapper] ESGOTOU TENTATIVAS em $(date)" >> "$LOG"
exit 1
