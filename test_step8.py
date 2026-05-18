"""Test Encoder — run: python test_step8.py"""
from model import Encoder, EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock
import torch
import torch.nn as nn

d_model = 512
h       = 8
d_ff    = 2048
dropout = 0.0
batch   = 2
seq_len = 5
N       = 6   # number of encoder layers

# Build N EncoderBlocks and wrap in Encoder
blocks = nn.ModuleList([
    EncoderBlock(
        d_model,
        MultiHeadAttentionBlock(d_model, h, dropout),
        FeedForwardBlock(d_model, d_ff, dropout),
        dropout
    )
    for _ in range(N)
])

encoder = Encoder(d_model, blocks)

# Fake input: 2 C++ programs, each 5 tokens, already embedded + positionally encoded
x = torch.randn(batch, seq_len, d_model)

# --- TEST 1: No mask ---
out = encoder(x, mask=None)
assert out.shape == (batch, seq_len, d_model)
print("TEST 1 PASSED: Encoder output shape correct")

# --- TEST 2: Output is normalised (final LayerNorm applied) ---
# Per-token mean should be ~0, std ~1
means = out.mean(dim=-1).abs().max().item()
assert means < 0.1
print(f"TEST 2 PASSED: Final LayerNorm applied — per-token mean ~0 (max={means:.6f})")

# --- TEST 3: With padding mask ---
src_mask             = torch.ones(batch, 1, 1, seq_len)
src_mask[1, 0, 0, 4] = 0   # program 2, token 4 is [PAD]
out_masked           = encoder(x, mask=src_mask)
assert out_masked.shape == (batch, seq_len, d_model)
assert not torch.equal(out, out_masked)
print("TEST 3 PASSED: Padding mask correctly changes encoder output")

# --- TEST 4: Total parameter count ---
total = sum(p.numel() for p in encoder.parameters())
print(f"\nTotal Encoder parameters: {total:,}")
print(f"  ({N} EncoderBlocks + 1 final LayerNorm)")
print("\nEncoder is ready. Its output will feed into the decoder as k and v in cross-attention.")
