"""Test ResidualConnection — run: python test_step6.py"""
from model import ResidualConnection, FeedForwardBlock, MultiHeadAttentionBlock
import torch

d_model = 512
dropout = 0.0
batch   = 2
seq_len = 5

residual = ResidualConnection(features=d_model, dropout=dropout)
x        = torch.randn(batch, seq_len, d_model)

# --- TEST 1: Wrapping FFN ---
ffn = FeedForwardBlock(d_model, d_ff=2048, dropout=dropout)
out = residual(x, sublayer=ffn)

assert out.shape == (batch, seq_len, d_model)
print("TEST 1 PASSED: ResidualConnection wrapping FFN — shape unchanged")

# --- TEST 2: Output is NOT equal to input (sublayer did something) ---
assert not torch.equal(out, x)
print("TEST 2 PASSED: Output differs from input — sublayer was applied")

# --- TEST 3: Output is NOT equal to just ffn(x) (residual was added) ---
ffn_only = ffn(residual.norm(x))
assert not torch.equal(out, ffn_only)
print("TEST 3 PASSED: Output differs from sublayer-only — residual +x was added")

# --- TEST 4: Verify the math: out == x + dropout(sublayer(norm(x))) ---
expected = x + ffn(residual.norm(x))   # dropout=0 so dropout is identity
assert torch.allclose(out, expected, atol=1e-5)
print("TEST 4 PASSED: out == x + sublayer(LayerNorm(x))  (math verified)")

# --- TEST 5: Wrapping MultiHeadAttention (self-attention) ---
mha      = MultiHeadAttentionBlock(d_model, h=8, dropout=dropout)
out_attn = residual(x, sublayer=lambda x: mha(x, x, x, mask=None))

assert out_attn.shape == (batch, seq_len, d_model)
print("TEST 5 PASSED: ResidualConnection wrapping MultiHeadAttention — shape unchanged")

print("\nSummary: one ResidualConnection class wraps both attention and FFN cleanly.")
