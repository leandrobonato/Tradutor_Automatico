#!/bin/bash
cd "D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico"
LOG="train_test_progress.log"
echo "[retry-wrapper] iniciado em $(date)" > "$LOG"
for i in $(seq 1 30); do
  echo "[retry-wrapper] tentativa $i em $(date)" >> "$LOG"
  python -m src.train --epochs 1 --max-train-examples 500 --max-val-examples 200 --batch-size 32 >> "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[retry-wrapper] SUCESSO em $(date)" >> "$LOG"
    exit 0
  fi
  echo "[retry-wrapper] falhou, aguardando 45s..." >> "$LOG"
  sleep 45
done
echo "[retry-wrapper] ESGOTOU TENTATIVAS em $(date)" >> "$LOG"
exit 1
