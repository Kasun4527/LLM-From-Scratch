"""
sample.py
=========
Load a trained checkpoint and generate text from the model.

Usage:
    python sample.py
    python sample.py --prompt "ROMEO:" --num 300
"""

import os
import pickle
import argparse

import torch

from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--prompt", type=str, default="\n",
                        help="Starting text to continue from.")
    parser.add_argument("--num", type=int, default=500, help="number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="<1 = safer/repetitive, >1 = more random")
    parser.add_argument("--top_k", type=int, default=200, help="sample only from top-k tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the vocabulary so we can encode the prompt and decode the output.
    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Load the checkpoint and rebuild the exact model that was trained.
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    print(f"loaded checkpoint from iter {ckpt['iter']} (val loss {ckpt['best_val_loss']:.4f})\n")

    # Encode the prompt into token ids, generate, then decode back to text.
    start_ids = encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # (1, T)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.num,
                           temperature=args.temperature, top_k=args.top_k)

    print(decode(y[0].tolist()))


if __name__ == "__main__":
    main()
