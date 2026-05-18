"""Test Dataset + DataLoader — run: python test_step14.py"""
from tokenizer import build_vocab, Vocabulary
from dataset import get_dataloader
from pathlib import Path
import torch

DATA      = Path("C:/Users/Debtirtha/Downloads/XLCoST_data/XLCoST_data/generation/pair_data_tok_full/C++-Python")
MAX_LEN   = 300
BATCH     = 8

# Build vocabularies
print("Building vocabularies...")
src_vocab = build_vocab(str(DATA / "train-C++-Python-tok.cpp"), min_freq=2)
tgt_vocab = build_vocab(str(DATA / "train-C++-Python-tok.py"),  min_freq=2)
print(f"C++ vocab: {len(src_vocab):,}  |  Python vocab: {len(tgt_vocab):,}\n")

# Build dataloader
train_loader = get_dataloader(
    str(DATA / "train-C++-Python-tok.cpp"),
    str(DATA / "train-C++-Python-tok.py"),
    src_vocab, tgt_vocab,
    max_seq_len=MAX_LEN, batch_size=BATCH, shuffle=True
)

# Grab one batch
src, tgt_input, tgt_output, src_mask, tgt_mask = next(iter(train_loader))

print(f"\n--- Batch shapes ---")
print(f"src:        {tuple(src.shape)}         (C++ tokens)")
print(f"tgt_input:  {tuple(tgt_input.shape)}         (Python decoder input)")
print(f"tgt_output: {tuple(tgt_output.shape)}         (Python labels to predict)")
print(f"src_mask:   {tuple(src_mask.shape)}")
print(f"tgt_mask:   {tuple(tgt_mask.shape)}")

# --- TEST 1: Shapes are correct ---
assert src.shape[0]       == BATCH
assert tgt_input.shape[0] == BATCH
assert src_mask.shape      == (BATCH, 1, 1, src.shape[1])
assert tgt_mask.shape      == (BATCH, 1, tgt_input.shape[1], tgt_input.shape[1])
print("\nTEST 1 PASSED: All batch shapes correct")

# --- TEST 2: tgt_input starts with [SOS], tgt_output ends with [EOS] ---
assert (tgt_input[:, 0] == Vocabulary.SOS_ID).all()
print("TEST 2 PASSED: All tgt_input sequences start with [SOS]")

# tgt_output last real token (before PAD) should be EOS
# Find the last non-PAD position per sequence
for i in range(BATCH):
    row       = tgt_output[i]
    real_toks = row[row != Vocabulary.PAD_ID]
    assert real_toks[-1].item() == Vocabulary.EOS_ID
print("TEST 3 PASSED: All tgt_output sequences end with [EOS]")

# --- TEST 4: tgt_input and tgt_output are offset by 1 (teacher forcing) ---
# For non-PAD tokens: tgt_input[i] should be tgt_output[i-1]
row_in  = tgt_input[0]
row_out = tgt_output[0]
assert row_in[1].item() == row_out[0].item()   # tgt_input[1] == tgt_output[0]
assert row_in[2].item() == row_out[1].item()   # tgt_input[2] == tgt_output[1]
print("TEST 4 PASSED: tgt_input and tgt_output are offset by 1 (teacher forcing verified)")

# --- TEST 5: Causal mask is lower triangular ---
tgt_len    = tgt_mask.shape[-1]
upper_vals = tgt_mask[0, 0][torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()]
assert not upper_vals.any()
print("TEST 5 PASSED: tgt_mask is causal (upper triangle is all False)")

# --- Show a decoded example ---
print("\n--- Example pair (program 0) ---")
print(f"C++:    {src_vocab.decode(src[0].tolist())[:80]}...")
print(f"Target: {tgt_vocab.decode(tgt_output[0].tolist())[:80]}...")

print(f"\nTotal batches in training set: {len(train_loader)}")
