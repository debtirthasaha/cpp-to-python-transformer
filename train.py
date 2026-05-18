"""
train.py — Training loop for C++ -> Python transformer.

Run: python train.py

Checkpoints are saved to checkpoints/ after every epoch if validation loss improves.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm

from tokenizer import build_vocab, Vocabulary
from dataset  import get_dataloader
from model    import build_transformer

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = Path("C:/Users/Debtirtha/Downloads/XLCoST_data/XLCoST_data/generation/pair_data_tok_full/C++-Python")
CKPT_DIR   = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# Model — smaller than paper defaults to fit GTX 1650 4GB
D_MODEL    = 256
N          = 4       # encoder + decoder layers each
H          = 8       # attention heads
D_FF       = 512     # feed-forward inner dim
DROPOUT    = 0.1

# Training
MAX_SEQ_LEN  = 300
BATCH_SIZE   = 8
EPOCHS       = 20
WARMUP_STEPS = 1000   # learning rate warmup steps (paper uses 4000, we scale down)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================
# From "Attention Is All You Need", section 5.3:
#   lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
#
# Phase 1 (step < warmup): lr increases linearly
# Phase 2 (step > warmup): lr decays proportional to 1/sqrt(step)
#
# WHY WARMUP?
#   At the start, weights are random. A large learning rate would
#   push them wildly in random directions. Warmup starts small and
#   gradually increases, letting the model find a good initial region
#   before stepping more aggressively.

def lr_lambda(step: int) -> float:
    # step starts at 0 — add 1 to avoid division by zero
    step = max(step, 1)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * WARMUP_STEPS ** -1.5)


# =============================================================================
# LOSS FUNCTION
# =============================================================================
# CrossEntropyLoss with ignore_index=PAD_ID:
#   - Expects raw logits (not softmax) — we call proj directly, not project()
#   - ignore_index tells it to skip [PAD] positions in the loss calculation
#     (we don't want the model penalised for what it predicts at padding positions)
#   - label_smoothing=0.1: instead of hard 0/1 targets, use 0.1/0.9
#     Prevents the model from being overconfident, improves generalisation.
#     The paper uses 0.1.

loss_fn = nn.CrossEntropyLoss(
    ignore_index   = Vocabulary.PAD_ID,
    label_smoothing = 0.1
)


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    loop = tqdm(loader, desc="  Training", leave=False)
    for src, tgt_input, tgt_output, src_mask, tgt_mask in loop:

        src        = src.to(device)
        tgt_input  = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        src_mask   = src_mask.to(device)
        tgt_mask   = tgt_mask.to(device)

        # --- Forward pass ---
        encoder_output = model.encode(src, src_mask)
        # (batch, src_len, d_model)

        decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        # (batch, tgt_len, d_model)

        logits = model.projection_layer.proj(decoder_output)
        # Raw logits — shape: (batch, tgt_len, vocab_size)
        # We call .proj directly (not .project()) to get raw logits for CrossEntropyLoss.

        # --- Compute loss ---
        # CrossEntropyLoss expects:
        #   input:  (N, C) where C = number of classes (vocab_size)
        #   target: (N,)   integer class indices
        # We reshape by merging batch and sequence dimensions.
        batch_size, tgt_len, vocab_size = logits.shape

        loss = loss_fn(
            logits.reshape(batch_size * tgt_len, vocab_size),
            tgt_output.reshape(batch_size * tgt_len)
        )

        # --- Backward pass ---
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: cap gradient norm at 1.0.
        # Prevents exploding gradients — a common problem in transformer training.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # --- Track loss (weighted by number of real tokens in batch) ---
        real_tokens  = (tgt_output != Vocabulary.PAD_ID).sum().item()
        total_loss  += loss.item() * real_tokens
        total_tokens += real_tokens

        loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

    return total_loss / total_tokens


# =============================================================================
# VALIDATE ONE EPOCH
# =============================================================================

def val_epoch(model, loader, device):
    model.eval()
    total_loss   = 0
    total_tokens = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="  Validation", leave=False)
        for src, tgt_input, tgt_output, src_mask, tgt_mask in loop:

            src        = src.to(device)
            tgt_input  = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            src_mask   = src_mask.to(device)
            tgt_mask   = tgt_mask.to(device)

            encoder_output = model.encode(src, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
            logits         = model.projection_layer.proj(decoder_output)

            batch_size, tgt_len, vocab_size = logits.shape
            loss = loss_fn(
                logits.reshape(batch_size * tgt_len, vocab_size),
                tgt_output.reshape(batch_size * tgt_len)
            )

            real_tokens  = (tgt_output != Vocabulary.PAD_ID).sum().item()
            total_loss  += loss.item() * real_tokens
            total_tokens += real_tokens

    return total_loss / total_tokens


# =============================================================================
# MAIN
# =============================================================================

def main():

    # --- Vocabularies ---
    print("Building vocabularies...")
    src_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.cpp"), min_freq=2)
    tgt_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.py"),  min_freq=2)
    print(f"C++ vocab: {len(src_vocab):,}  |  Python vocab: {len(tgt_vocab):,}")

    # --- DataLoaders ---
    print("\nBuilding dataloaders...")
    train_loader = get_dataloader(
        str(DATA_DIR / "train-C++-Python-tok.cpp"),
        str(DATA_DIR / "train-C++-Python-tok.py"),
        src_vocab, tgt_vocab,
        max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        str(DATA_DIR / "val-C++-Python-tok.cpp"),
        str(DATA_DIR / "val-C++-Python-tok.py"),
        src_vocab, tgt_vocab,
        max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Model ---
    print("\nBuilding model...")
    model = build_transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        src_seq_len    = MAX_SEQ_LEN,
        tgt_seq_len    = MAX_SEQ_LEN,
        d_model        = D_MODEL,
        N              = N,
        h              = H,
        dropout        = DROPOUT,
        d_ff           = D_FF,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Optimizer + Scheduler ---
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # lr=1.0 because the lr_lambda function returns the actual learning rate.
    # betas and eps are from the paper (section 5.3).

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Training loop ---
    best_val_loss = float('inf')
    print(f"\nStarting training for {EPOCHS} epochs on {DEVICE}...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_loss   = val_epoch(model, val_loader, DEVICE)

        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = CKPT_DIR / "best_model.pt"
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':       val_loss,
                'src_vocab':      src_vocab,
                'tgt_vocab':      tgt_vocab,
            }, ckpt_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        print()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {CKPT_DIR / 'best_model.pt'}")


if __name__ == '__main__':
    main()
