"""
dataset.py — Dataset and DataLoader for C++ -> Python translation.

Each training example produces three sequences:
  src        : C++ tokens + [EOS]               -> encoder reads this
  tgt_input  : [SOS] + Python tokens            -> decoder reads this (teacher forcing)
  tgt_output : Python tokens + [EOS]            -> what the model must predict

TEACHER FORCING:
  At training step i, the decoder input is the CORRECT previous token,
  not the model's own prediction. This is faster and more stable to train.

  Example for "x = 5":
    tgt_input  : [SOS]  x    =    5
    tgt_output :  x     =    5   [EOS]
    position:      0    1    2    3
  At position 0, decoder sees [SOS] and must predict "x".
  At position 1, decoder sees "x"   and must predict "=".
  At position 2, decoder sees "="   and must predict "5".
  At position 3, decoder sees "5"   and must predict [EOS].
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizer import Vocabulary


# =============================================================================
# DATASET
# =============================================================================

class CodeTranslationDataset(Dataset):

    def __init__(self, cpp_file: str, py_file: str,
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 max_seq_len: int):
        # max_seq_len: pairs where either sequence exceeds this are skipped.

        self.src_vocab   = src_vocab
        self.tgt_vocab   = tgt_vocab
        self.max_seq_len = max_seq_len
        self.pairs       = []   # list of (src_ids, tgt_input_ids, tgt_output_ids)

        skipped = 0
        with open(cpp_file, 'r', encoding='utf-8') as f_src, \
             open(py_file,  'r', encoding='utf-8') as f_tgt:

            for src_line, tgt_line in zip(f_src, f_tgt):

                # Encoder input: C++ tokens + [EOS]
                # [EOS] tells the encoder "the source sequence ends here"
                src_ids = src_vocab.encode(src_line, add_sos=False, add_eos=True)

                # Full Python sequence with both [SOS] and [EOS]
                full_tgt_ids = tgt_vocab.encode(tgt_line, add_sos=True, add_eos=True)

                # Decoder INPUT  = [SOS] + Python tokens (drop last [EOS])
                # The decoder reads this one step at a time.
                tgt_input_ids = full_tgt_ids[:-1]

                # Decoder OUTPUT = Python tokens + [EOS]  (drop first [SOS])
                # This is the ground truth the model is trained to predict.
                tgt_output_ids = full_tgt_ids[1:]

                # Skip pairs that are too long for our seq_len budget
                if len(src_ids) > max_seq_len or len(tgt_input_ids) > max_seq_len:
                    skipped += 1
                    continue

                self.pairs.append((
                    torch.tensor(src_ids,        dtype=torch.long),
                    torch.tensor(tgt_input_ids,  dtype=torch.long),
                    torch.tensor(tgt_output_ids, dtype=torch.long),
                ))

        print(f"Loaded {len(self.pairs):,} pairs  ({skipped:,} skipped — too long)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# =============================================================================
# COLLATE FUNCTION
# =============================================================================
# PyTorch's DataLoader calls this function on each batch.
# Its job: pad all sequences in the batch to the same length,
#          then build the src_mask and tgt_mask tensors.

def collate_fn(batch):
    """
    batch: list of (src_ids, tgt_input_ids, tgt_output_ids) tuples.
    All tensors in a batch must be the same length — we pad with PAD_ID=0.
    """

    src_ids_list        = [item[0] for item in batch]
    tgt_input_ids_list  = [item[1] for item in batch]
    tgt_output_ids_list = [item[2] for item in batch]

    # pad_sequence stacks tensors and pads shorter ones with padding_value.
    # batch_first=True -> output shape: (batch, max_len_in_batch)
    src        = pad_sequence(src_ids_list,        batch_first=True, padding_value=Vocabulary.PAD_ID)
    tgt_input  = pad_sequence(tgt_input_ids_list,  batch_first=True, padding_value=Vocabulary.PAD_ID)
    tgt_output = pad_sequence(tgt_output_ids_list, batch_first=True, padding_value=Vocabulary.PAD_ID)
    # Dynamic padding: pad only to the longest sequence IN THIS BATCH,
    # not to max_seq_len. Saves memory and speeds up training.

    # --- Source mask (padding mask) ---
    # Shape: (batch, 1, 1, src_len)
    # 1 where token is real, 0 where it is [PAD].
    # The (1, 1) dims allow broadcasting over (h, query_seq_len) in attention.
    src_mask = (src != Vocabulary.PAD_ID).unsqueeze(1).unsqueeze(1)
    # src != PAD_ID : (batch, src_len) bool
    # .unsqueeze(1)  : (batch, 1, src_len)
    # .unsqueeze(1)  : (batch, 1, 1, src_len)

    # --- Target mask (causal + padding) ---
    # Shape: (batch, 1, tgt_len, tgt_len)
    # Combines two masks:
    #   1. Padding mask:  zeros out [PAD] tokens in the target
    #   2. Causal mask:   token i cannot attend to token j > i (no peeking ahead)

    tgt_len = tgt_input.size(1)

    # Padding mask: (batch, 1, 1, tgt_len) — 1 for real, 0 for PAD
    tgt_pad_mask = (tgt_input != Vocabulary.PAD_ID).unsqueeze(1).unsqueeze(1)

    # Causal mask: (1, 1, tgt_len, tgt_len) — lower triangular
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    # torch.tril: keeps lower triangle, zeros upper triangle
    # Position (i, j): 1 if j <= i (token i can see token j), 0 otherwise

    # Combine: both conditions must be true
    tgt_mask = tgt_pad_mask & causal_mask
    # (batch, 1, 1, tgt_len) & (1, 1, tgt_len, tgt_len) = (batch, 1, tgt_len, tgt_len)

    return src, tgt_input, tgt_output, src_mask, tgt_mask


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_dataloader(cpp_file: str, py_file: str,
                   src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                   max_seq_len: int, batch_size: int,
                   shuffle: bool = True) -> DataLoader:

    dataset = CodeTranslationDataset(
        cpp_file, py_file, src_vocab, tgt_vocab, max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
        num_workers = 0,   # 0 = load in main process (safe on Windows)
    )
