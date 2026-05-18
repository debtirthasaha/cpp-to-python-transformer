"""
inference.py — Greedy decoding for C++ -> Python translation.

GREEDY DECODING:
  At each step, pick the single most likely next token (argmax).
  Simple and fast — not optimal (beam search is better) but good enough
  to evaluate whether the model has learned to translate.

  Step 0: decoder input = [SOS]
  Step 1: run decoder -> argmax -> token_1 ; decoder input = [SOS, token_1]
  Step 2: run decoder -> argmax -> token_2 ; decoder input = [SOS, token_1, token_2]
  ...
  Stop when: model predicts [EOS], or we hit max_seq_len.

Usage:
  python inference.py                     # interactive mode
  python inference.py --file input.cpp    # translate a file
"""

import re
import torch
import argparse
from pathlib import Path
from tokenizer import Vocabulary


# =============================================================================
# RAW C++ TOKENIZER
# =============================================================================

# Regex: order matters — longer patterns must come before shorter ones.
_TOKEN_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"'                                        # string literal
    r"|'(?:[^'\\]|\\.)*'"                                       # char literal
    r'|<<=|>>='                                                  # 3-char operators
    r'|<=|>=|==|!=|\+\+|--|<<|>>|->|::|&&|\|\|'                # 2-char operators
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^='                            # compound assignment
    r'|[(){}\[\];,.]'                                            # punctuation
    r'|[+\-*/%=<>!&|^~?:#]'                                    # single-char operators
    r'|\w+'                                                      # identifiers / numbers
    r'|\S'                                                       # catch-all
)

def tokenize_raw_cpp(code: str) -> str:
    """
    Convert raw C++ source to XLCoST-style space-tokenized format.

    Every operator, punctuation mark, and identifier is separated by spaces.
    Newlines become NEW_LINE tokens.  Comments are stripped.

    Example:
        int x = 5; return x;
        -> 'int x = 5 ; NEW_LINE return x ; NEW_LINE'
    """
    # Strip comments
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)

    tokens = []
    for line in code.split('\n'):
        line = line.strip()
        if not line:
            continue
        tokens.extend(_TOKEN_RE.findall(line))
        tokens.append('NEW_LINE')

    return ' '.join(tokens)


# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

def load_model(ckpt_path: str, device: torch.device):
    """Load model + vocabularies from a saved checkpoint."""
    from model import build_transformer

    ckpt = torch.load(ckpt_path, map_location=device)

    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']

    model = build_transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        src_seq_len    = 300,
        tgt_seq_len    = 300,
        d_model        = 256,
        N              = 4,
        h              = 8,
        dropout        = 0.0,   # no dropout at inference
        d_ff           = 512,
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")
    return model, src_vocab, tgt_vocab


# =============================================================================
# GREEDY DECODE
# =============================================================================

def greedy_decode(model, src_ids: torch.Tensor, src_mask: torch.Tensor,
                  tgt_vocab: Vocabulary, max_len: int, device: torch.device) -> list:
    """
    Generate a Python token sequence from encoded C++ input.

    src_ids:  (1, src_len) — C++ token IDs (batch size 1)
    src_mask: (1, 1, 1, src_len) — padding mask
    Returns:  list of generated token IDs (excluding [SOS], including [EOS])
    """

    # --- Encode the C++ source ONCE ---
    encoder_output = model.encode(src_ids, src_mask)
    # (1, src_len, d_model)
    # This is computed once and reused at every decoder step.
    # This is why encode() and decode() are separate methods.

    # --- Initialise decoder input with [SOS] ---
    decoder_input = torch.tensor([[Vocabulary.SOS_ID]], dtype=torch.long, device=device)
    # Shape: (1, 1) — batch=1, one token so far

    generated = []

    for _ in range(max_len):

        tgt_len = decoder_input.shape[1]

        # Causal mask for current decoder input length
        tgt_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device)
        ).unsqueeze(0).unsqueeze(0)
        # (1, 1, tgt_len, tgt_len)

        # --- Run decoder ---
        decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
        # (1, tgt_len, d_model)

        # --- Project last position to vocabulary ---
        logits     = model.projection_layer.proj(decoder_output[:, -1, :])
        # Only the LAST position matters — that's the next token prediction.
        # (1, vocab_size)

        next_token = logits.argmax(dim=-1).item()
        # Greedy: pick the highest-scoring token.

        # --- Stop if [EOS] ---
        if next_token == Vocabulary.EOS_ID:
            break

        generated.append(next_token)

        # --- Append predicted token to decoder input for next step ---
        decoder_input = torch.cat([
            decoder_input,
            torch.tensor([[next_token]], dtype=torch.long, device=device)
        ], dim=1)
        # (1, tgt_len + 1)

    return generated


# =============================================================================
# TRANSLATE
# =============================================================================

def translate(cpp_code: str, model, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
              device: torch.device, max_len: int = 300) -> str:
    """
    Translate a space-tokenized C++ string to Python.

    cpp_code: already space-tokenized, e.g. "int x = 5 ; return x ;"
    Returns:  space-tokenized Python string
    """
    model.eval()
    with torch.no_grad():

        # Encode C++ input
        src_ids = src_vocab.encode(cpp_code, add_sos=False, add_eos=True)

        if len(src_ids) > max_len:
            print(f"Warning: input length {len(src_ids)} exceeds max_len {max_len}. Truncating.")
            src_ids = src_ids[:max_len - 1] + [Vocabulary.EOS_ID]

        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        # (1, src_len)

        src_mask = (src_tensor != Vocabulary.PAD_ID).unsqueeze(1).unsqueeze(1)
        # (1, 1, 1, src_len)

        # Greedy decode
        output_ids = greedy_decode(model, src_tensor, src_mask, tgt_vocab, max_len, device)

        # Decode to text
        return tgt_vocab.decode(output_ids, skip_special=True)


# =============================================================================
# POST-PROCESS
# =============================================================================

def postprocess(text: str) -> str:
    """
    Convert XLCoST's tokenized Python back to readable Python code.

    XLCoST uses:
      NEW_LINE -> actual newline
      INDENT   -> increase indentation
      DEDENT   -> decrease indentation
    """
    tokens = text.split()
    result = []
    indent = 0

    for token in tokens:
        if token == 'NEW_LINE':
            result.append('\n' + '    ' * indent)
        elif token == 'INDENT':
            indent += 1
        elif token == 'DEDENT':
            indent = max(0, indent - 1)
        else:
            result.append(token + ' ')

    return ''.join(result).strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='C++ -> Python translator')
    parser.add_argument('--ckpt',    default='checkpoints/best_model.pt', help='checkpoint path')
    parser.add_argument('--file',    default=None, help='C++ file to translate')
    parser.add_argument('--raw',     action='store_true', help='input is raw C++ (auto-tokenizes)')
    parser.add_argument('--max_len', type=int, default=300)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not Path(args.ckpt).exists():
        print(f"No checkpoint found at {args.ckpt}. Train the model first with: python train.py")
        return

    model, src_vocab, tgt_vocab = load_model(args.ckpt, device)

    if args.file:
        # Translate from file — read whole file if --raw, else line-by-line
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()

        if args.raw:
            cpp_tok = tokenize_raw_cpp(content)
            print("C++ (tokenized):")
            print(cpp_tok[:200] + ("..." if len(cpp_tok) > 200 else ""))
            raw = translate(cpp_tok, model, src_vocab, tgt_vocab, device, args.max_len)
            print("\nPython:")
            print(postprocess(raw))
        else:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw = translate(line, model, src_vocab, tgt_vocab, device, args.max_len)
                print("Python:")
                print(postprocess(raw))
                print("\n" + "="*60 + "\n")
    else:
        # Interactive mode
        mode = "raw C++" if args.raw else "pre-tokenized C++"
        print(f"\nInteractive C++ -> Python translator  [{mode} mode]")
        if args.raw:
            print("Paste raw C++ code, then press Enter twice to translate. Ctrl+C to quit.\n")
        else:
            print("Paste a tokenized C++ line and press Enter. Ctrl+C to quit.\n")

        while True:
            try:
                if args.raw:
                    print("C++> (paste code, blank line to submit)")
                    lines = []
                    while True:
                        line = input()
                        if line == "":
                            break
                        lines.append(line)
                    cpp_input = "\n".join(lines).strip()
                    if not cpp_input:
                        continue
                    cpp_input = tokenize_raw_cpp(cpp_input)
                else:
                    cpp_input = input("C++ tokens> ").strip()
                    if not cpp_input:
                        continue

                raw = translate(cpp_input, model, src_vocab, tgt_vocab, device, args.max_len)
                print("\nPython:")
                print(postprocess(raw))
                print()
            except KeyboardInterrupt:
                print("\nDone.")
                break


if __name__ == '__main__':
    main()
