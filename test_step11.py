"""Test ProjectionLayer — run: python test_step11.py"""
from model import ProjectionLayer
import torch

d_model    = 512
vocab_size = 8000   # Python vocabulary size
batch      = 2
tgt_len    = 4

proj = ProjectionLayer(d_model, vocab_size)

# Fake decoder output
x   = torch.randn(batch, tgt_len, d_model)
out = proj(x)

print(f"Input  shape: {x.shape}")
print(f"Output shape: {out.shape}")

# --- TEST 1: Output shape is (batch, tgt_len, vocab_size) ---
assert out.shape == (batch, tgt_len, vocab_size)
print("\nTEST 1 PASSED: Output shape (batch, tgt_len, vocab_size)")

# --- TEST 2: log_softmax — all values <= 0 ---
assert (out <= 0).all()
print("TEST 2 PASSED: All log-probabilities <= 0")

# --- TEST 3: exp(log_softmax) sums to 1.0 per position (valid distribution) ---
probs    = out.exp()
row_sums = probs.sum(dim=-1)   # sum over vocab_size
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
print("TEST 3 PASSED: exp(log_softmax) sums to 1.0 — valid probability distribution")

# --- TEST 4: argmax gives the predicted next token ---
# At inference we only care about the LAST position (the next token to generate)
last_pos_logprobs = out[:, -1, :]              # (batch, vocab_size)
predicted_tokens  = last_pos_logprobs.argmax(dim=-1)   # (batch,)
assert predicted_tokens.shape == (batch,)
print(f"TEST 4 PASSED: Predicted next tokens: {predicted_tokens.tolist()}")
print(f"  (token IDs from Python vocabulary of size {vocab_size})")

# --- Parameter count ---
total = sum(p.numel() for p in proj.parameters())
print(f"\nProjectionLayer parameters: {total:,}")
print(f"  = vocab_size ({vocab_size}) x d_model ({d_model}) = {vocab_size * d_model:,}")
