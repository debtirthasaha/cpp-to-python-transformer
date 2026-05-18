"""Test Decoder — run: python test_step10.py"""
from model import (Decoder, DecoderBlock, Encoder, EncoderBlock,
                   MultiHeadAttentionBlock, FeedForwardBlock)
import torch
import torch.nn as nn

d_model = 512
h       = 8
d_ff    = 2048
dropout = 0.0
batch   = 2
src_len = 5   # C++ sequence length
tgt_len = 4   # Python sequence length
N       = 6   # number of decoder layers

# --- Build Decoder ---
decoder = Decoder(
    d_model,
    nn.ModuleList([
        DecoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout
        )
        for _ in range(N)
    ])
)

# Fake inputs
encoder_output = torch.randn(batch, src_len, d_model)
x              = torch.randn(batch, tgt_len, d_model)
causal_mask    = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)

# --- TEST 1: Output shape correct ---
out = decoder(x, encoder_output, src_mask=None, tgt_mask=causal_mask)
assert out.shape == (batch, tgt_len, d_model)
print("TEST 1 PASSED: Decoder output shape correct (batch, tgt_len, d_model)")

# --- TEST 2: Final LayerNorm applied ---
means = out.mean(dim=-1).abs().max().item()
assert means < 0.1
print(f"TEST 2 PASSED: Final LayerNorm applied — per-token mean ~0 (max={means:.6f})")

# --- TEST 3: Different encoder output -> different decoder output ---
encoder_output_2 = torch.randn(batch, src_len, d_model)
out2 = decoder(x, encoder_output_2, src_mask=None, tgt_mask=causal_mask)
assert not torch.equal(out, out2)
print("TEST 3 PASSED: Decoder output changes with different C++ encoder input")

# --- TEST 4: Full Encoder -> Decoder pipeline ---
encoder = Encoder(
    d_model,
    nn.ModuleList([
        EncoderBlock(d_model,
                     MultiHeadAttentionBlock(d_model, h, dropout),
                     FeedForwardBlock(d_model, d_ff, dropout),
                     dropout)
        for _ in range(N)
    ])
)

cpp_tokens = torch.randn(batch, src_len, d_model)
py_tokens  = torch.randn(batch, tgt_len, d_model)

enc_out  = encoder(cpp_tokens, mask=None)
dec_out  = decoder(py_tokens, enc_out, src_mask=None, tgt_mask=causal_mask)

assert dec_out.shape == (batch, tgt_len, d_model)
print("TEST 4 PASSED: Full Encoder -> Decoder pipeline works")

# --- Parameter count ---
total = sum(p.numel() for p in decoder.parameters())
print(f"\nTotal Decoder parameters: {total:,}")
print(f"  ({N} DecoderBlocks + 1 final LayerNorm)")
print("\nDecoder output is ready to feed into the ProjectionLayer (Step 11).")
