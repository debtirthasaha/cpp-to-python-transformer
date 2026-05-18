"""Test PositionalEncoding — run: python test_step2.py"""
from model import InputEmbeddings, PositionalEncoding
import torch

# --- Setup ---
vocab_size = 1000
d_model    = 512
seq_len    = 10    # max sequence length
dropout    = 0.0   # 0 so values aren't randomly zeroed during our checks

embeddings = InputEmbeddings(d_model, vocab_size)
pos_enc    = PositionalEncoding(d_model, seq_len, dropout)

# --- Fake input: 2 programs, each 5 tokens ---
x = torch.tensor([
    [34, 102, 8, 55, 11],
    [22,  67, 90, 14, 38]
])

embedded = embeddings(x)             # shape: (2, 5, 512)
output   = pos_enc(embedded)         # shape: (2, 5, 512)

print(f"After embedding shape:          {embedded.shape}")
print(f"After positional encoding shape: {output.shape}")

# --- TEST 1: Shape is unchanged ---
assert output.shape == embedded.shape, "Shape changed — something is wrong!"
print("\nTEST 1 PASSED: Shape unchanged after positional encoding")

# --- TEST 2: Values actually changed (position was added) ---
assert not torch.equal(embedded, output), "Values didn't change — PE wasn't added!"
print("TEST 2 PASSED: Values changed — positional encoding was added")

# --- TEST 3: Same token at different positions → different final vectors ---
# Make a sequence where token 34 appears at position 0 AND position 3
x2 = torch.tensor([[34, 5, 5, 34, 5]])   # token 34 at pos 0 and pos 3
emb2 = embeddings(x2)
out2 = pos_enc(emb2)

pos0_vec = out2[0, 0, :]   # token 34 at position 0
pos3_vec = out2[0, 3, :]   # token 34 at position 3

assert not torch.equal(pos0_vec, pos3_vec), "Same token at different positions should differ!"
print("TEST 3 PASSED: Same token at different positions -> different vectors")

# --- Show the difference ---
print(f"\nToken 34 at position 0 (first 6 values): {pos0_vec[:6].tolist()}")
print(f"Token 34 at position 3 (first 6 values): {pos3_vec[:6].tolist()}")
print("\n-> The model can now distinguish position 0 from position 3!")
