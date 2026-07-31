#!/bin/bash
cd "D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico"
LOG="preprocess_progress.log"
echo "[retry-wrapper] iniciado em $(date)" > "$LOG"
for i in $(seq 1 40); do
  echo "[retry-wrapper] tentativa $i em $(date)" >> "$LOG"
  python -m src.data.preprocess >> "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[retry-wrapper] SUCESSO em $(date)" >> "$LOG"
    exit 0
  fi
  echo "[retry-wrapper] falhou, aguardando 30s..." >> "$LOG"
  sleep 30
done
echo "[retry-wrapper] ESGOTOU TENTATIVAS em $(date)" >> "$LOG"
exit 1
