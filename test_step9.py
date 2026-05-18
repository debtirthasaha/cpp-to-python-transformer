"""Test DecoderBlock — run: python test_step9.py"""
from model import (DecoderBlock, EncoderBlock, Encoder,
                   MultiHeadAttentionBlock, FeedForwardBlock)
import torch
import torch.nn as nn

d_model  = 512
h        = 8
d_ff     = 2048
dropout  = 0.0
batch    = 2
src_len  = 5   # C++ sequence length
tgt_len  = 4   # Python sequence length (can differ from src_len)

# --- Build a DecoderBlock ---
decoder_block = DecoderBlock(
    d_model,
    self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout),
    cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout),
    feed_forward_block    = FeedForwardBlock(d_model, d_ff, dropout),
    dropout               = dropout
)

# Fake inputs
encoder_output = torch.randn(batch, src_len, d_model)  # C++ context from encoder
x              = torch.randn(batch, tgt_len, d_model)  # Python tokens so far

# --- TEST 1: No masks — output shape correct ---
out = decoder_block(x, encoder_output, src_mask=None, tgt_mask=None)
assert out.shape == (batch, tgt_len, d_model)
print("TEST 1 PASSED: Output shape correct (batch, tgt_len, d_model)")

# --- TEST 2: src and tgt sequence lengths can differ ---
assert out.shape[1] == tgt_len   # output follows TARGET length, not source
print(f"TEST 2 PASSED: Output length = tgt_len={tgt_len}, not src_len={src_len}")

# --- TEST 3: Causal mask — token i cannot attend to token j > i ---
causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)
out_masked  = decoder_block(x, encoder_output, src_mask=None, tgt_mask=causal_mask)
assert out_masked.shape == (batch, tgt_len, d_model)
assert not torch.equal(out, out_masked)
print("TEST 3 PASSED: Causal tgt_mask changes output (future tokens blocked)")

# --- TEST 4: Different encoder_output changes the result (cross-attention works) ---
encoder_output_2 = torch.randn(batch, src_len, d_model)  # different C++ context
out2 = decoder_block(x, encoder_output_2, src_mask=None, tgt_mask=causal_mask)
assert not torch.equal(out_masked, out2)
print("TEST 4 PASSED: Different encoder output changes decoder output (cross-attention works)")

# --- TEST 5: Full pipeline — Encoder -> DecoderBlock ---
# Build a small encoder
encoder_blocks = nn.ModuleList([
    EncoderBlock(d_model,
                 MultiHeadAttentionBlock(d_model, h, dropout),
                 FeedForwardBlock(d_model, d_ff, dropout),
                 dropout)
    for _ in range(6)
])
from model import Encoder
encoder     = Encoder(d_model, encoder_blocks)
cpp_input   = torch.randn(batch, src_len, d_model)
enc_out     = encoder(cpp_input, mask=None)        # C++ -> encoder output
dec_out     = decoder_block(x, enc_out, src_mask=None, tgt_mask=causal_mask)
assert dec_out.shape == (batch, tgt_len, d_model)
print("TEST 5 PASSED: Full Encoder -> DecoderBlock pipeline works")

total = sum(p.numel() for p in decoder_block.parameters())
print(f"\nParameters per DecoderBlock: {total:,}")
print("(More than EncoderBlock because of the extra cross-attention sublayer)")
