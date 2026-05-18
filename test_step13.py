"""Test tokenizer — run: python test_step13.py"""
from tokenizer import Vocabulary, build_vocab, dataset_stats
from pathlib import Path

DATA = Path("C:/Users/Debtirtha/Downloads/XLCoST_data/XLCoST_data/generation/pair_data_tok_full/C++-Python")
CPP_TRAIN = DATA / "train-C++-Python-tok.cpp"
PY_TRAIN  = DATA / "train-C++-Python-tok.py"

# --- TEST 1: Build vocabularies from training data ---
print("Building vocabularies...")
src_vocab = build_vocab(str(CPP_TRAIN), min_freq=2)
tgt_vocab = build_vocab(str(PY_TRAIN),  min_freq=2)

print(f"\nC++    vocabulary size: {len(src_vocab):,} tokens")
print(f"Python vocabulary size: {len(tgt_vocab):,} tokens")
assert len(src_vocab) > 4   # more than just special tokens
assert len(tgt_vocab) > 4
print("TEST 1 PASSED: Vocabularies built")

# --- TEST 2: Special tokens are at correct IDs ---
assert src_vocab.token_to_id['[PAD]'] == 0
assert src_vocab.token_to_id['[UNK]'] == 1
assert src_vocab.token_to_id['[SOS]'] == 2
assert src_vocab.token_to_id['[EOS]'] == 3
print("\nTEST 2 PASSED: Special tokens at correct IDs (PAD=0, UNK=1, SOS=2, EOS=3)")

# --- TEST 3: encode() and decode() are inverse operations ---
sample = "int x = 5 ; return x ;"
ids    = src_vocab.encode(sample)
decoded = src_vocab.decode(ids)
assert decoded == sample
print(f"TEST 3 PASSED: encode -> decode roundtrip")
print(f"  '{sample}'")
print(f"  -> {ids}")
print(f"  -> '{decoded}'")

# --- TEST 4: SOS and EOS added correctly ---
ids_with_sos_eos = tgt_vocab.encode("x = 5", add_sos=True, add_eos=True)
assert ids_with_sos_eos[0]  == Vocabulary.SOS_ID
assert ids_with_sos_eos[-1] == Vocabulary.EOS_ID
print(f"\nTEST 4 PASSED: SOS prepended, EOS appended")
print(f"  'x = 5' -> {ids_with_sos_eos}")

# --- TEST 5: Unknown token handling ---
ids_with_unk = src_vocab.encode("xyzzy_unknown_token_12345")
assert ids_with_unk[0] == Vocabulary.UNK_ID
print(f"\nTEST 5 PASSED: Unknown tokens map to UNK_ID={Vocabulary.UNK_ID}")

# --- TEST 6: XLCoST special tokens exist in vocab ---
for special in ['NEW_LINE', 'INDENT', 'DEDENT']:
    assert special in tgt_vocab.token_to_id, f"{special} missing from Python vocab!"
print(f"\nTEST 6 PASSED: XLCoST special tokens found in Python vocab (NEW_LINE, INDENT, DEDENT)")

# --- Dataset stats ---
print("\n--- Dataset Statistics (max_seq_len=150) ---")
dataset_stats(str(CPP_TRAIN), str(PY_TRAIN), src_vocab, tgt_vocab, max_seq_len=150)
