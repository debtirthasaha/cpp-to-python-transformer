"""Test inference pipeline — run: python test_step16.py"""
import torch
from tokenizer import build_vocab, Vocabulary
from model     import build_transformer
from inference import greedy_decode, translate, postprocess
from pathlib   import Path

DATA_DIR = Path("C:/Users/Debtirtha/Downloads/XLCoST_data/XLCoST_data/generation/pair_data_tok_full/C++-Python")
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build vocab + untrained model (to test the pipeline mechanics)
src_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.cpp"), min_freq=2)
tgt_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.py"),  min_freq=2)

model = build_transformer(
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    src_seq_len=300, tgt_seq_len=300,
    d_model=256, N=4, h=8, dropout=0.0, d_ff=512
).to(DEVICE)
model.eval()

# A real C++ line from the training data
cpp_input = "int x = 5 ; return x ;"

# --- TEST 1: translate() returns a string ---
output = translate(cpp_input, model, src_vocab, tgt_vocab, DEVICE, max_len=50)
assert isinstance(output, str)
print("TEST 1 PASSED: translate() returns a string")
print(f"  Input  (C++):   {cpp_input}")
print(f"  Output (Python): {output[:80]}")

# --- TEST 2: output contains only Python vocabulary tokens ---
for tok in output.split():
    assert tok in tgt_vocab.token_to_id or tok == '[UNK]', f"Unknown token in output: {tok}"
print("TEST 2 PASSED: All output tokens are in Python vocabulary")

# --- TEST 3: greedy_decode stops at EOS or max_len ---
src_ids    = src_vocab.encode(cpp_input, add_sos=False, add_eos=True)
src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
src_mask   = (src_tensor != Vocabulary.PAD_ID).unsqueeze(1).unsqueeze(1)
generated  = greedy_decode(model, src_tensor, src_mask, tgt_vocab, max_len=20, device=DEVICE)
assert len(generated) <= 20
assert Vocabulary.EOS_ID not in generated   # EOS is consumed as stop signal, not returned
print(f"TEST 3 PASSED: greedy_decode respects max_len (generated {len(generated)} tokens)")

# --- TEST 4: postprocess converts NEW_LINE / INDENT / DEDENT correctly ---
# INDENT before a NEW_LINE so indentation appears mid-string, not stripped by .strip()
raw     = "def f ( ) : NEW_LINE INDENT return x NEW_LINE DEDENT pass"
fmt     = postprocess(raw)
assert '\n' in fmt
assert '    ' in fmt   # indentation appears mid-string before 'return'
print("TEST 4 PASSED: postprocess() converts NEW_LINE/INDENT/DEDENT correctly")
print(f"  Raw:       {raw}")
print(f"  Formatted:\n{fmt}")

# --- Show a real example from validation set ---
print("\n--- Real example from validation set ---")
with open(DATA_DIR / "val-C++-Python-tok.cpp", encoding='utf-8') as f:
    cpp_line = f.readline().strip()
with open(DATA_DIR / "val-C++-Python-tok.py", encoding='utf-8') as f:
    py_line = f.readline().strip()

pred = translate(cpp_line, model, src_vocab, tgt_vocab, DEVICE, max_len=150)

print(f"C++ (first 80 chars):     {cpp_line[:80]}...")
print(f"Ground truth (first 80):  {py_line[:80]}...")
print(f"Prediction (first 80):    {pred[:80]}...")
print("\n(Prediction is gibberish — model is untrained. Run python train.py to fix that!)")
