# Blog Idea: C++ → Python Transformer From Scratch

## Potential titles
- "I trained a C++→Python code translator from scratch on a $200 GPU — here's what I learned"
- "Transformers from scratch: not English→French, but C++→Python"

---

## What makes this original (vs existing blogs)

1. **Code translation, not text** — Every tutorial uses English→French. XLCoST + competitive programming = different problem space.
2. **Consumer GPU constraints** — Real decisions made because of GTX 1650 4GB (d_model=256, dynamic padding, VRAM math).
3. **Vocabulary coverage analysis** — Why Hello World fails but binary search works (min_freq=2, UNK analysis).
4. **Pre-tokenization gap** — XLCoST ships pre-tokenized; had to build a raw C++ tokenizer to bridge the gap.
5. **Real overfitting data** — Actual epoch-by-epoch train vs val loss numbers.

---

## Real training numbers to include

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 13    | 1.9109    | 2.0615   |
| 15    | 1.8708    | 2.0545   |
| 16    | 1.8542    | 2.0511   |
| 19    | 1.8103    | **2.0474** ← best checkpoint |
| 20    | 1.7964    | 2.0576   ← starts overfitting |

Model: 16.4M params, d_model=256, N=4, h=8, d_ff=512, dropout=0.1, label_smoothing=0.1

---

## Suggested structure

1. **The problem** — Code translation with transformers, XLCoST dataset
2. **Architecture decisions** — Why encoder-decoder, every hyperparam explained for 4GB GPU
3. **The dataset problem** — XLCoST pre-tokenized format, NEW_LINE/INDENT/DEDENT, the raw tokenizer built to fix it
4. **Training: what the numbers mean** — Real loss curves, warmup LR schedule, label smoothing
5. **What the model actually learned** — Binary search result, Hello World failure, attention failure modes
6. **Honest verdict** — val_loss=2.047 in context, limitations, what would make it better

---

## Future addition (do before publishing)
- Attention visualization: use saved `attention_scores` in MultiHeadAttentionBlock to show which
  C++ tokens attend to which Python tokens during translation. No one has this for code translation.
  Would make a great shareable figure.

---

## Project location
C:/Users/Debtirtha/Desktop/transformer_project/

## Key files
- model.py       — all 12 transformer components
- train.py       — training loop
- inference.py   — greedy decode + raw C++ tokenizer
- checkpoints/best_model.pt  — epoch 19, val_loss=2.0474
