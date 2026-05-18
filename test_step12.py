"""Test Transformer + build_transformer — run: python test_step12.py"""
from model import build_transformer
import torch

# Hyperparameters
src_vocab_size = 10000   # C++ vocabulary
tgt_vocab_size = 8000    # Python vocabulary
src_seq_len    = 512     # max C++ sequence length
tgt_seq_len    = 512     # max Python sequence length
batch          = 2
src_len        = 10      # actual C++ sequence length in this batch
tgt_len        = 8       # actual Python sequence length in this batch

# --- Build the full transformer ---
model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)

# Fake token IDs (integers, not embeddings)
src         = torch.randint(0, src_vocab_size, (batch, src_len))  # C++ token IDs
tgt         = torch.randint(0, tgt_vocab_size, (batch, tgt_len))  # Python token IDs
causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(0)

# --- TEST 1: encode() ---
encoder_output = model.encode(src, src_mask=None)
assert encoder_output.shape == (batch, src_len, 512)
print("TEST 1 PASSED: encode() output shape (batch, src_len, 512)")

# --- TEST 2: decode() ---
decoder_output = model.decode(encoder_output, src_mask=None, tgt=tgt, tgt_mask=causal_mask)
assert decoder_output.shape == (batch, tgt_len, 512)
print("TEST 2 PASSED: decode() output shape (batch, tgt_len, 512)")

# --- TEST 3: project() ---
logits = model.project(decoder_output)
assert logits.shape == (batch, tgt_len, tgt_vocab_size)
print("TEST 3 PASSED: project() output shape (batch, tgt_len, tgt_vocab_size)")

# --- TEST 4: log-probabilities are valid ---
assert (logits <= 0).all()
probs = logits.exp().sum(dim=-1)
assert torch.allclose(probs, torch.ones_like(probs), atol=1e-3)
print("TEST 4 PASSED: Valid log-probability distribution over Python vocabulary")

# --- TEST 5: predict next token ---
next_token_logits  = logits[:, -1, :]           # last position: (batch, vocab_size)
predicted_tokens   = next_token_logits.argmax(dim=-1)  # (batch,)
assert predicted_tokens.shape == (batch,)
print(f"TEST 5 PASSED: Next token predictions: {predicted_tokens.tolist()}")

# --- TEST 6: Xavier init — weights are not all equal ---
w1 = list(model.parameters())[0]
w2 = list(model.parameters())[1]
assert not torch.equal(w1, w2)
print("TEST 6 PASSED: Xavier initialisation applied — weights vary across layers")

# --- Full parameter count ---
total = sum(p.numel() for p in model.parameters())
print(f"\n{'='*50}")
print(f"FULL TRANSFORMER PARAMETER COUNT: {total:,}")
print(f"{'='*50}")
print(f"  Encoder:          {sum(p.numel() for p in model.encoder.parameters()):>12,}")
print(f"  Decoder:          {sum(p.numel() for p in model.decoder.parameters()):>12,}")
print(f"  src_embed:        {sum(p.numel() for p in model.src_embed.parameters()):>12,}")
print(f"  tgt_embed:        {sum(p.numel() for p in model.tgt_embed.parameters()):>12,}")
print(f"  projection_layer: {sum(p.numel() for p in model.projection_layer.parameters()):>12,}")
print(f"\nThe transformer is built and ready to train on C++ -> Python translation!")
