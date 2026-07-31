#!/bin/bash
cd "D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico"
export HF_HOME="D:/Repositórios/Portfolio/Deep Learning/Tradutor_Automatico/data/raw/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME"
LOG="download_progress.log"
echo "[retry-wrapper] iniciado em $(date)" > "$LOG"
for i in $(seq 1 40); do
  echo "[retry-wrapper] tentativa $i em $(date)" >> "$LOG"
  python -m src.data.download >> "$LOG" 2>&1
  if [ $? -eq 0 ]; then
    echo "[retry-wrapper] SUCESSO em $(date)" >> "$LOG"
    exit 0
  fi
  echo "[retry-wrapper] falhou, aguardando 30s..." >> "$LOG"
  sleep 30
done
echo "[retry-wrapper] ESGOTOU TENTATIVAS em $(date)" >> "$LOG"
exit 1
