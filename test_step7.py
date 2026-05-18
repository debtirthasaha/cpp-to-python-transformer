"""Test EncoderBlock — run: python test_step7.py"""
from model import EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock
import torch

d_model = 512
h       = 8
d_ff    = 2048
dropout = 0.0
batch   = 2
seq_len = 5

# Build sub-components first, then inject into EncoderBlock
self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
feed_forward   = FeedForwardBlock(d_model, d_ff, dropout)
encoder_block  = EncoderBlock(d_model, self_attention, feed_forward, dropout)

# Fake input: 2 C++ programs, each 5 tokens, already embedded
x = torch.randn(batch, seq_len, d_model)

# --- TEST 1: No mask ---
out = encoder_block(x, src_mask=None)
assert out.shape == (batch, seq_len, d_model)
print("TEST 1 PASSED: Output shape correct with no mask")

# --- TEST 2: With padding mask ---
# Simulate: C++ program 2 is shorter — token 4 is [PAD], should be ignored
src_mask             = torch.ones(batch, 1, 1, seq_len)
src_mask[1, 0, 0, 4] = 0   # mask out token 4 for program 2
out_masked           = encoder_block(x, src_mask=src_mask)
assert out_masked.shape == (batch, seq_len, d_model)
print("TEST 2 PASSED: Output shape correct with padding mask")

# --- TEST 3: Masking changes the output ---
assert not torch.equal(out, out_masked)
print("TEST 3 PASSED: Mask changes output (masked [PAD] token ignored in attention)")

# --- TEST 4: Stack 6 EncoderBlocks (full encoder depth) ---
blocks = []
for _ in range(6):
    attn = MultiHeadAttentionBlock(d_model, h, dropout)
    ffn  = FeedForwardBlock(d_model, d_ff, dropout)
    blocks.append(EncoderBlock(d_model, attn, ffn, dropout))

out6 = x
for block in blocks:
    out6 = block(out6, src_mask=None)

assert out6.shape == (batch, seq_len, d_model)
print("TEST 4 PASSED: 6 stacked EncoderBlocks output shape unchanged")

# --- Show parameter count ---
total = sum(p.numel() for p in encoder_block.parameters())
print(f"\nParameters per EncoderBlock: {total:,}")
print(f"Parameters in 6-block stack: {total * 6:,}")
print("\nEach C++ token now carries context from all other tokens in the sequence.")
print("Ready to be used by the Python-generating decoder via cross-attention.")
