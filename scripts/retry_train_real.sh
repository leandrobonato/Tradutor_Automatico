#!/bin/bash
cd "D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico"
LOG="train_real_progress.log"
echo "[retry-wrapper] iniciado em $(date)" > "$LOG"
for i in $(seq 1 60); do
  echo "[retry-wrapper] tentativa $i em $(date)" >> "$LOG"
  python -m src.train --epochs 4 --max-train-examples 4000 --max-val-examples 400 --batch-size 16 >> "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[retry-wrapper] SUCESSO em $(date)" >> "$LOG"
    exit 0
  fi
  echo "[retry-wrapper] falhou, aguardando 30s..." >> "$LOG"
  sleep 30
done
echo "[retry-wrapper] ESGOTOU TENTATIVAS em $(date)" >> "$LOG"
exit 1
