"""
prepare_data.py
===============
Turns raw text into the binary token files the trainer reads.

We use a simple CHARACTER-LEVEL tokenizer here: every distinct character becomes
one token. This is the easiest tokenizer to understand and is perfect for the
laptop sanity-check. (Later, on the HPC, you can swap in a BPE tokenizer like
tiktoken for real efficiency -- see the note at the bottom.)

Outputs (into the data/ folder):
    train.bin  -- training tokens as uint16
    val.bin    -- validation tokens as uint16
    meta.pkl   -- the vocabulary mappings (so sample.py can decode back to text)

Usage:
    python prepare_data.py
    python prepare_data.py --input path/to/your.txt
"""

import os
import pickle
import argparse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main(input_path=None):
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Get the raw text. If no file is given and none exists, download the classic
    #    "tiny shakespeare" dataset so you have something to train on immediately.
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "input.txt")
        if not os.path.exists(input_path):
            print("No input.txt found -- downloading tiny shakespeare sample...")
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # 2) Build the vocabulary: the sorted set of unique characters.
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"vocab size: {vocab_size}")

    # 3) Create the two lookup tables (string<->int).
    stoi = {ch: i for i, ch in enumerate(chars)}   # char  -> token id
    itos = {i: ch for i, ch in enumerate(chars)}   # token id -> char

    def encode(s):
        return [stoi[c] for c in s]

    # 4) Split 90% train / 10% validation.
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val   has {len(val_ids):,} tokens")

    # 5) Save the binary token files.
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    # 6) Save the vocabulary so we can decode generated tokens back to characters.
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\nDone. Files written to: {DATA_DIR}")
    print("Next: python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a .txt file. If omitted, downloads tiny shakespeare.")
    args = parser.parse_args()
    main(args.input)


# -----------------------------------------------------------------------------
# NOTE on tokenization for later (HPC stage):
# Character-level is great for learning but inefficient -- the model wastes capacity
# learning to spell. Real LLMs use sub-word tokenizers (BPE). To upgrade:
#   pip install tiktoken
#   enc = tiktoken.get_encoding("gpt2")     # ~50k token vocab
#   ids = enc.encode_ordinary(text)
# Then set vocab_size=50257 in your training config and skip meta.pkl (use tiktoken
# to decode in sample.py instead).
# -----------------------------------------------------------------------------
