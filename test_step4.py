"""Test FeedForwardBlock — run: python test_step4.py"""
from model import FeedForwardBlock
import torch

d_model = 512
d_ff    = 2048   # 4 * d_model, as in the paper
dropout = 0.0    # 0 so values aren't randomly zeroed during checks
batch   = 2
seq_len = 5

ffn = FeedForwardBlock(d_model, d_ff, dropout)

x      = torch.randn(batch, seq_len, d_model)
output = ffn(x)

print(f"Input  shape: {x.shape}")
print(f"Output shape: {output.shape}")

# --- TEST 1: Shape is unchanged (512 -> 2048 -> 512) ---
assert output.shape == (batch, seq_len, d_model)
print("\nTEST 1 PASSED: Shape unchanged (512 -> 2048 -> 512)")

# --- TEST 2: Parameters exist with correct sizes ---
params = dict(ffn.named_parameters())
assert params['linear_1.weight'].shape == (d_ff, d_model),    "linear_1 weight wrong shape"
assert params['linear_1.bias'].shape   == (d_ff,),            "linear_1 bias wrong shape"
assert params['linear_2.weight'].shape == (d_model, d_ff),    "linear_2 weight wrong shape"
assert params['linear_2.bias'].shape   == (d_model,),         "linear_2 bias wrong shape"
print("TEST 2 PASSED: Parameter shapes correct")
print(f"  linear_1 weight: {params['linear_1.weight'].shape}  (d_ff x d_model)")
print(f"  linear_2 weight: {params['linear_2.weight'].shape}  (d_model x d_ff)")

# --- TEST 3: Total parameter count ---
total_params = sum(p.numel() for p in ffn.parameters())
expected     = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
assert total_params == expected
print(f"\nTEST 3 PASSED: Parameter count = {total_params:,}")
print(f"  linear_1: {d_model} x {d_ff} weights + {d_ff} biases = {d_model*d_ff + d_ff:,}")
print(f"  linear_2: {d_ff} x {d_model} weights + {d_model} biases = {d_ff*d_model + d_model:,}")

# --- TEST 4: Each token is processed independently ---
# If we change token 0 of program 1, only that token's output should change.
x2        = x.clone()
x2[0, 0]  = torch.randn(d_model)   # change only token 0 of program 1

out2      = ffn(x2)

changed   = not torch.equal(output[0, 0], out2[0, 0])   # token 0 changed
unchanged = torch.equal(output[0, 1], out2[0, 1])        # token 1 unchanged

assert changed and unchanged
print("\nTEST 4 PASSED: Tokens are processed independently")
print("  Token 0 changed  -> output changed")
print("  Token 1 unchanged -> output unchanged")
