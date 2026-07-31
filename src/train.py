"""Treino do modelo Seq2Seq com Atenção de Bahdanau.

Uso:
    python -m src.train --epochs 10 --batch-size 64
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.data.dataset import TranslationDataset, collate_fn
from src.data.tokenizer import load_tokenizer
from src.models.seq2seq import Seq2Seq


def run_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches = 0

    for src, src_lens, tgt, _tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0)
            # outputs: (batch, tgt_len-1, vocab) alinhado com tgt[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo Seq2Seq EN->PT.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--teacher-forcing", type=float, default=config.TEACHER_FORCING_RATIO)
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="Limita o nº de exemplos de treino (útil para execução rápida em CPU).")
    parser.add_argument("--max-val-examples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] dispositivo: {device}")

    sp_en = load_tokenizer(config.SRC_LANG)
    sp_pt = load_tokenizer(config.TGT_LANG)

    train_ds = TranslationDataset("train", sp_en, sp_pt)
    val_ds = TranslationDataset("val", sp_en, sp_pt)

    if args.max_train_examples:
        train_ds.pairs = train_ds.pairs[: args.max_train_examples]
    if args.max_val_examples:
        val_ds.pairs = val_ds.pairs[: args.max_val_examples]

    print(f"[train] exemplos de treino: {len(train_ds)} | validação: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(src_vocab_size=sp_en.get_piece_size(), tgt_vocab_size=sp_pt.get_piece_size()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parâmetros treináveis: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_ID)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.MODELS_DIR / "seq2seq_best.pt"
    history_path = config.REPORTS_DIR / "training_history.json"

    best_val_loss = float("inf")
    # Retoma de um checkpoint anterior (mesmos vocabulários), caso exista —
    # torna o treino resiliente a interrupções por falta de memória, comuns
    # neste ambiente com RAM fortemente disputada por outros processos.
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        if (state.get("src_vocab_size") == sp_en.get_piece_size()
                and state.get("tgt_vocab_size") == sp_pt.get_piece_size()):
            model.load_state_dict(state["model_state_dict"])
            best_val_loss = state.get("val_loss", float("inf"))
            print(f"[train] retomando de checkpoint existente (val_loss={best_val_loss:.4f})")

    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        for key in ("train_loss", "val_loss", "epoch_seconds"):
            history.setdefault(key, [])
    else:
        history = {"train_loss": [], "val_loss": [], "epoch_seconds": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, args.teacher_forcing, train=False)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_seconds"].append(elapsed)

        print(f"[train] epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "src_vocab_size": sp_en.get_piece_size(),
                "tgt_vocab_size": sp_pt.get_piece_size(),
            }, checkpoint_path)
            print(f"[train] novo melhor checkpoint salvo em {checkpoint_path} (val_loss={val_loss:.4f})")

        # Salva o histórico a cada época (não só ao final) para não perder
        # progresso caso o processo seja interrompido por falta de memória.
        history["n_train_examples"] = len(train_ds)
        history["n_val_examples"] = len(val_ds)
        history["n_params"] = n_params
        history["best_val_loss"] = best_val_loss
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[train] histórico salvo em {history_path}")


if __name__ == "__main__":
    main()
