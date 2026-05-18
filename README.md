# C++ → Python Transformer

A 16.4M-parameter encoder-decoder transformer for C++ → Python code translation. Trained on XLCoST on a GTX 1650 (4 GB). Best checkpoint at epoch 19, val_loss 2.0474.

## Architecture

- 4 encoder layers + 4 decoder layers
- d_model 256, 8 heads, d_ff 512
- Pre-norm, sinusoidal positional encoding
- Dropout 0.1, label smoothing 0.1, Adam with warmup + 1/sqrt(step) decay
- Greedy decoding at inference

All 12 components live in `model.py`. Built and tested incrementally — `test_step1.py` through `test_step16.py` verify each component in isolation.

## Files

- `model.py` — all 12 transformer components (Embeddings, PositionalEncoding, LayerNorm, FFN, MultiHeadAttention, ResidualConnection, EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer, Transformer).
- `tokenizer.py` — vocabulary + space-tokenizer for the XLCoST format.
- `dataset.py` — PyTorch Dataset for loading parallel C++/Python pairs.
- `train.py` — training loop.
- `inference.py` — greedy decode, plus a raw C++ tokenizer so you can pipe in real source code.
- `learned_snapshot.py` — load a trained checkpoint and dump embedding neighbors + attention weights.
- `test_step*.py` — 16 numbered tests, one per component.
- `BLOG_IDEA.md` — original blog draft.

## Trained checkpoint

The `checkpoints/best_model.pt` file is 189 MB — too large for GitHub. Download it from Hugging Face: **[debtirthasaha/cpp-to-python-transformer](https://huggingface.co/debtirthasaha/cpp-to-python-transformer)**

Place it at `checkpoints/best_model.pt`, then:

```bash
python inference.py --raw     # interactive mode, accepts raw C++
```

## Dataset

Trained on [XLCoST](https://github.com/reddy-lab-code-research/XLCoST) (Reddy Lab's parallel competitive-programming corpus). Pre-tokenized: every operator and identifier is space-separated, with `NEW_LINE`, `INDENT`, `DEDENT` for whitespace.

## Writeup

[A transformer that reads C++ and writes Python](https://debtirthasaha.github.io/blog/2026/cpp-to-python-transformer/)
