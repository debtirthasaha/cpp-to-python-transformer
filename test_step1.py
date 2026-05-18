"""Test InputEmbeddings — run: python test_step1.py"""
from model import InputEmbeddings
import torch

# --- Setup ---
vocab_size = 1000   # pretend our vocabulary has 1000 unique code tokens
d_model = 512       # each token becomes a 512-dimensional vector

embeddings = InputEmbeddings(d_model, vocab_size)

# --- Fake input: 2 C++ programs, each 5 tokens ---
x = torch.tensor([
    [34, 102, 8, 55, 11],   # "int x = 5 ;"
    [22, 67, 90, 14, 38]    # "float y = 3 ;"
])

print(f"Input shape:  {x.shape}")        # (2, 5)
output = embeddings(x)
print(f"Output shape: {output.shape}")   # (2, 5, 512)
print(f"\nFirst token embedding (first 8 values):")
print(output[0, 0, :8])

# --- Same token ID = same embedding ---
x2 = torch.tensor([[34, 34]])
out2 = embeddings(x2)
print(f"\nSame token 34 at pos 0 and pos 1 — identical? {torch.equal(out2[0,0], out2[0,1])}")
