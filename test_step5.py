"""Test MultiHeadAttentionBlock — run: python test_step5.py"""
from model import MultiHeadAttentionBlock
import torch

d_model = 512
h       = 8
dropout = 0.0
batch   = 2
seq_len = 5

mha = MultiHeadAttentionBlock(d_model, h, dropout)

x    = torch.randn(batch, seq_len, d_model)
out  = mha(x, x, x, mask=None)   # self-attention: q=k=v=x

print(f"Input  shape: {x.shape}")
print(f"Output shape: {out.shape}")

# --- TEST 1: Output shape unchanged ---
assert out.shape == (batch, seq_len, d_model)
print("\nTEST 1 PASSED: Output shape unchanged (batch, seq_len, d_model)")

# --- TEST 2: Attention scores saved, shape is (batch, h, seq_len, seq_len) ---
scores = mha.attention_scores
assert scores.shape == (batch, h, seq_len, seq_len)
print(f"TEST 2 PASSED: Attention scores shape {scores.shape}")
print(f"  = (batch={batch}, heads={h}, seq_len={seq_len}, seq_len={seq_len})")

# --- TEST 3: Attention weights sum to 1 per row (they are probabilities) ---
row_sums = scores.sum(dim=-1)   # sum over the "which key" dimension
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
print("TEST 3 PASSED: Attention weights sum to 1.0 per row (valid probability distribution)")

# --- TEST 4: Causal mask — token i cannot attend to token j > i ---
# Build a lower-triangular mask: 1 = allowed, 0 = blocked
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
# shape: (1, 1, seq_len, seq_len) — broadcasts over batch and heads

out_masked = mha(x, x, x, mask=causal_mask)
scores_masked = mha.attention_scores   # (batch, h, seq_len, seq_len)

# Upper triangle (future positions) should have ~0 attention weight
upper_triangle = scores_masked[0, 0][torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()]
assert upper_triangle.abs().max().item() < 1e-6
print("TEST 4 PASSED: Causal mask works — future positions have ~0 attention weight")

# --- TEST 5: Cross-attention — q from one source, k and v from another ---
encoder_out = torch.randn(batch, seq_len, d_model)
decoder_out = torch.randn(batch, seq_len, d_model)
cross_out   = mha(decoder_out, encoder_out, encoder_out, mask=None)
assert cross_out.shape == (batch, seq_len, d_model)
print("TEST 5 PASSED: Cross-attention works (q=decoder, k=v=encoder)")

# --- Show attention pattern for head 0 ---
print(f"\nAttention weights for head 0, program 0 (rows=queries, cols=keys):")
print(torch.round(scores_masked[0, 0] * 100) / 100)
print("Each row sums to 1.0. Upper triangle is 0 (masked future).")
