"""Test LayerNormalization — run: python test_step3.py"""
from model import LayerNormalization
import torch

d_model = 512
batch   = 2
seq_len = 5

layer_norm = LayerNormalization(features=d_model)

# --- Fake input with extreme values (like real activations after attention) ---
x = torch.randn(batch, seq_len, d_model) * 100   # wild scale: values in [-300, 300]

output = layer_norm(x)

print(f"Input  shape: {x.shape}")
print(f"Output shape: {output.shape}")

# --- TEST 1: Shape unchanged ---
assert output.shape == x.shape
print("\nTEST 1 PASSED: Shape unchanged")

# --- TEST 2: Mean per token is ~0 after normalization ---
# For each token, compute mean across 512 dims — should be near 0
token_means = output.mean(dim=-1)   # shape: (2, 5)
max_mean = token_means.abs().max().item()
assert max_mean < 0.1, f"Mean too large: {max_mean}"
print(f"TEST 2 PASSED: Per-token mean ~0 (max abs mean = {max_mean:.6f})")

# --- TEST 3: Std per token is ~1 after normalization ---
token_stds = output.std(dim=-1)    # shape: (2, 5)
max_std_err = (token_stds - 1.0).abs().max().item()
assert max_std_err < 0.1, f"Std too far from 1: {max_std_err}"
print(f"TEST 3 PASSED: Per-token std ~1  (max deviation from 1 = {max_std_err:.6f})")

# --- TEST 4: alpha and beta are learnable parameters ---
param_names = [name for name, _ in layer_norm.named_parameters()]
assert 'alpha' in param_names and 'bias' in param_names
print(f"TEST 4 PASSED: Learnable parameters found: {param_names}")

# --- TEST 5: Input values were wild, output is tame ---
print(f"\nInput  — mean: {x.mean().item():.2f},  std: {x.std().item():.2f}")
print(f"Output — mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
print("\n-> Layer norm tamed the wild activations!")
