"""
learned_snapshot.py — Dump what the trained C++->Python transformer learned.

Loads checkpoints/best_model.pt and produces:
  1. Nearest neighbors of selected source-side (C++) tokens in embedding space
  2. Nearest neighbors of selected target-side (Python) tokens
  3. Attention head 0 weights for the encoder's last layer on one tiny example
"""

import torch, math, os, sys, io

# Force UTF-8 stdout so Windows doesn't blow up on box-drawing characters.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pt')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device('cpu')

from model import build_transformer
from tokenizer import Vocabulary

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
src_vocab = ckpt['src_vocab']
tgt_vocab = ckpt['tgt_vocab']

print(f'epoch:    {ckpt["epoch"]}')
print(f'val loss: {ckpt["val_loss"]:.4f}')
print(f'src vocab size: {len(src_vocab):,}')
print(f'tgt vocab size: {len(tgt_vocab):,}')

model = build_transformer(
    src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
    src_seq_len=300, tgt_seq_len=300,
    d_model=256, N=4, h=8, dropout=0.0, d_ff=512,
).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()


# ---- 1 & 2 — embedding nearest neighbors ----

def nearest(token, vocab, embed_layer, k=8):
    if token not in vocab.token_to_id:
        return None
    W = embed_layer.embedding.weight.detach()           # (V, d_model)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-12)
    q = W[vocab.token_to_id[token]]
    sims = W @ q
    top = torch.topk(sims, k=k + 1)
    out = []
    for s, idx in zip(top.values.tolist(), top.indices.tolist()):
        tok = vocab.id_to_token[idx]
        if tok == token: continue
        out.append((tok, s))
    return out[:k]

src_seeds = ['int', 'for', 'if', 'vector', '+', '==', 'cout', '<', 'true', 'string']
tgt_seeds = ['def', 'for', 'if', 'range', '+', '==', 'print', '<', 'True', 'str']

print('\n--- SOURCE (C++) embedding nearest neighbors ---')
for t in src_seeds:
    nn = nearest(t, src_vocab, model.src_embed)
    if nn is None:
        print(f'  {t!r}: not in vocab')
        continue
    print(f'  {t!r:>10} -> ' + ', '.join(f'{tok}({s:.2f})' for tok, s in nn[:6]))

print('\n--- TARGET (Python) embedding nearest neighbors ---')
for t in tgt_seeds:
    nn = nearest(t, tgt_vocab, model.tgt_embed)
    if nn is None:
        print(f'  {t!r}: not in vocab')
        continue
    print(f'  {t!r:>10} -> ' + ', '.join(f'{tok}({s:.2f})' for tok, s in nn[:6]))


# ---- 3 — attention on one example ----

example_cpp = 'int sum = a + b ; NEW_LINE return sum ;'
src_ids = torch.tensor([src_vocab.encode(example_cpp, add_eos=True)], dtype=torch.long, device=DEVICE)
src_tokens = src_vocab.decode(src_ids[0].tolist(), skip_special=False).split()

with torch.no_grad():
    src_mask = torch.ones(1, 1, 1, src_ids.size(1), device=DEVICE)
    enc_out = model.encode(src_ids, src_mask)

last_block = model.encoder.layers[-1]
attn_block = last_block.self_attention_block
attn = attn_block.attention_scores         # (B, h, T, T)
if attn is None:
    print('\n[!] no attention_scores buffer found on the last encoder block.')
else:
    attn = attn[0]                          # (h, T, T)
    h0 = attn[0]                            # head 0
    T = h0.size(0)
    print(f'\n--- encoder self-attention, head 0, last layer ---')
    print(f'input: {" ".join(src_tokens)}')
    print(f'shape: {tuple(h0.shape)}  (T x T)')

    header = '       ' + ''.join(f'{t[:6]:>7}' for t in src_tokens)
    print(header)
    for i, t in enumerate(src_tokens):
        row = ''.join(f'{h0[i, j].item():>7.2f}' for j in range(T))
        print(f'{t[:6]:>6} {row}')

    # Also: for each query position, which key position got max attention
    print(f'\n--- argmax-key per query (head 0) ---')
    for i, t in enumerate(src_tokens):
        j = int(h0[i].argmax().item())
        print(f'  q[{i:2d}] {t:>10} -> k[{j:2d}] {src_tokens[j]:<10}  ({h0[i, j].item():.2f})')
