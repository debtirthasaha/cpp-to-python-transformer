"""
tokenizer.py — Vocabulary builder and tokenizer for C++ -> Python translation.

The XLCoST data is already space-tokenized, so our tokenizer is simple:
  text  -> split by spaces -> look up each token in vocab -> list of IDs
  IDs   -> look up each ID in vocab -> join by spaces -> text

Special tokens:
  [PAD] = 0  padding to equalise sequence lengths in a batch
  [UNK] = 1  unknown token (not seen in training vocabulary)
  [SOS] = 2  start of sequence (prepended to every target/decoder input)
  [EOS] = 3  end of sequence   (appended to every target, signals stop)
"""

from collections import Counter
from pathlib import Path


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:

    # Reserved special token IDs — these four slots are always taken.
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3

    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[SOS]', '[EOS]']

    def __init__(self):
        # token -> ID
        self.token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        # ID -> token
        self.id_to_token = {i: tok for i, tok in enumerate(self.SPECIAL_TOKENS)}

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary if not already present. Returns its ID."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx]   = token
        return self.token_to_id[token]

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, text: str, add_sos=False, add_eos=False):
        """
        Convert a space-tokenized string to a list of token IDs.
        Unknown tokens are mapped to UNK_ID.

        add_sos: prepend [SOS] — used for decoder INPUT  (teacher forcing)
        add_eos: append  [EOS] — used for decoder TARGET (what we want to predict)
        """
        tokens = text.strip().split()
        ids    = [self.token_to_id.get(t, self.UNK_ID) for t in tokens]
        if add_sos:
            ids = [self.SOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids, skip_special=True):
        """
        Convert a list of token IDs back to a space-separated string.
        skip_special: remove [PAD], [SOS], [EOS] from output (keep [UNK] for debugging).
        """
        skip = {self.PAD_ID, self.SOS_ID, self.EOS_ID} if skip_special else set()
        tokens = [self.id_to_token.get(i, '[UNK]') for i in ids if i not in skip]
        return ' '.join(tokens)


# =============================================================================
# VOCABULARY BUILDER
# =============================================================================

def build_vocab(filepath: str, min_freq: int = 2) -> Vocabulary:
    """
    Build a Vocabulary by scanning a tokenized text file.
    Each line is one program. Tokens are space-separated.

    min_freq: tokens appearing fewer than this many times are excluded
              (they'll map to [UNK] at encode time).
              Reduces vocab size and handles typos/rare symbols.
    """
    counter = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    vocab = Vocabulary()
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab.add_token(token)

    return vocab


# =============================================================================
# DATASET STATS HELPER
# =============================================================================

def dataset_stats(cpp_file: str, py_file: str, src_vocab: Vocabulary,
                  tgt_vocab: Vocabulary, max_seq_len: int):
    """
    Print statistics about the dataset and show how many pairs
    will be kept vs filtered out by max_seq_len.
    """
    total = kept = 0
    src_lengths = []
    tgt_lengths = []

    with open(cpp_file, 'r', encoding='utf-8') as f_src, \
         open(py_file,  'r', encoding='utf-8') as f_tgt:

        for src_line, tgt_line in zip(f_src, f_tgt):
            src_ids = src_vocab.encode(src_line, add_sos=False, add_eos=True)
            tgt_ids = tgt_vocab.encode(tgt_line, add_sos=True,  add_eos=True)
            total += 1
            src_lengths.append(len(src_ids))
            tgt_lengths.append(len(tgt_ids))
            if len(src_ids) <= max_seq_len and len(tgt_ids) <= max_seq_len:
                kept += 1

    avg_src = sum(src_lengths) / len(src_lengths)
    avg_tgt = sum(tgt_lengths) / len(tgt_lengths)
    max_src = max(src_lengths)
    max_tgt = max(tgt_lengths)

    print(f"Total pairs:          {total:,}")
    print(f"Kept (len<={max_seq_len}):     {kept:,}  ({100*kept/total:.1f}%)")
    print(f"Filtered out:         {total-kept:,}  ({100*(total-kept)/total:.1f}%)")
    print(f"Avg C++    length:    {avg_src:.1f} tokens")
    print(f"Avg Python length:    {avg_tgt:.1f} tokens")
    print(f"Max C++    length:    {max_src} tokens")
    print(f"Max Python length:    {max_tgt} tokens")
