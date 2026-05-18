"""Sanity check train loop — run: python test_step15.py"""
import torch
import torch.nn as nn
from tokenizer import build_vocab, Vocabulary
from dataset   import get_dataloader
from model     import build_transformer
from pathlib   import Path

DATA_DIR = Path("C:/Users/Debtirtha/Downloads/XLCoST_data/XLCoST_data/generation/pair_data_tok_full/C++-Python")
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}\n")

# Build vocab + loader
src_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.cpp"), min_freq=2)
tgt_vocab = build_vocab(str(DATA_DIR / "train-C++-Python-tok.py"),  min_freq=2)

loader = get_dataloader(
    str(DATA_DIR / "train-C++-Python-tok.cpp"),
    str(DATA_DIR / "train-C++-Python-tok.py"),
    src_vocab, tgt_vocab, max_seq_len=300, batch_size=8, shuffle=False
)

# Build small model
model = build_transformer(
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    src_seq_len=300, tgt_seq_len=300,
    d_model=256, N=4, h=8, d_ff=512, dropout=0.1
).to(DEVICE)

loss_fn   = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_ID, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# One forward + backward pass
src, tgt_input, tgt_output, src_mask, tgt_mask = next(iter(loader))
src, tgt_input, tgt_output = src.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)
src_mask, tgt_mask         = src_mask.to(DEVICE), tgt_mask.to(DEVICE)

model.train()
enc_out = model.encode(src, src_mask)
dec_out = model.decode(enc_out, src_mask, tgt_input, tgt_mask)
logits  = model.projection_layer.proj(dec_out)

b, t, v = logits.shape
loss = loss_fn(logits.reshape(b*t, v), tgt_output.reshape(b*t))
loss.backward()
optimizer.step()

print(f"TEST PASSED: Forward + backward pass successful")
print(f"  logits shape:  {tuple(logits.shape)}")
print(f"  loss value:    {loss.item():.4f}  (expect ~log({len(tgt_vocab)})={__import__('math').log(len(tgt_vocab)):.2f} for random weights)")
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check VRAM if on GPU
if DEVICE.type == 'cuda':
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM used: {used:.2f} GB / {total:.1f} GB  ({100*used/total:.1f}%)")

print("\nAll good — run python train.py to start training!")
