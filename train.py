"""
train.py
========
The training loop. This is where the model actually LEARNS by:
  1) grabbing a random batch of text
  2) predicting the next token at every position
  3) measuring how wrong it was (the loss)
  4) nudging the weights to be a little less wrong (backprop + optimizer step)
  ...repeated thousands of times.

Defaults are tuned to RUN ON A LAPTOP CPU in a few minutes for a sanity check.
Scale-up settings for the RTX 5090 are documented in README.md.

Usage:
    python prepare_data.py     # once, to create data/*.bin
    python train.py
"""

import os
import time
import math
import pickle

import numpy as np
import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Configuration -- edit these. (Small = laptop sanity check.)
# -----------------------------------------------------------------------------
out_dir = "out"                # where checkpoints are saved
data_dir = "data"              # where train.bin / val.bin / meta.pkl live

# How long to train and how often to check in.
max_iters = 2000               # total training steps (laptop test). HPC: 100000+
eval_interval = 250            # how often to evaluate on val set
eval_iters = 50                # how many batches to average for an eval estimate
log_interval = 10              # how often to print the training loss

# Model size (small for laptop). HPC: bump n_layer/n_head/n_embd/block_size way up.
block_size = 128
batch_size = 16
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Optimizer.
learning_rate = 1e-3           # small models like a higher LR
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.99
grad_clip = 1.0                # clip gradients to this norm (training stability)

# Learning-rate schedule (warmup then cosine decay). Simple and effective.
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-4

# System.
device = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on modern GPUs (incl. RTX 5090) is fast and stable; fall back to fp32 on CPU.
dtype = "bfloat16" if (device == "cuda" and torch.cuda.is_bf16_supported()) else "float32"
seed = 1337
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
os.makedirs(out_dir, exist_ok=True)
device_type = "cuda" if "cuda" in device else "cpu"
print(f"device={device}, dtype={dtype}")

# Load the vocab size from meta.pkl (created by prepare_data.py).
meta_path = os.path.join(data_dir, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
print(f"vocab_size = {vocab_size}")

# Memory-map the token files so we don't load everything into RAM at once.
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    """Grab a random batch of (input, target) pairs.

    For each sample we pick a random starting point, take `block_size` tokens as the
    input x, and the SAME tokens shifted right by one as the target y. So the model
    learns: given tokens[0..i], predict token[i+1].
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Build the model.
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, vocab_size=vocab_size,
                  dropout=dropout, bias=True)
model = GPT(GPTConfig(**model_args))
model.to(device)

# Mixed-precision context + gradient scaler (only matters on CUDA).
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
       if device_type == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False))
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)


@torch.no_grad()
def estimate_loss():
    """Average the loss over a few batches of train AND val so we can watch for
    overfitting (val loss creeping up while train loss keeps dropping)."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """Warmup linearly, then cosine-decay down to min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# The training loop
# -----------------------------------------------------------------------------
print("starting training...")
best_val_loss = float("inf")
t0 = time.time()

for it in range(max_iters + 1):
    # Set the learning rate for this step.
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Periodically evaluate and save the best checkpoint.
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "model_args": model_args,
                "iter": it,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            print(f"  -> saved checkpoint to {out_dir}/ckpt.pt")

    # --- one training step ---
    X, Y = get_batch("train")
    with ctx:
        _, loss = model(X, Y)        # forward pass: compute loss
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()    # backward pass: compute gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)           # update the weights
    scaler.update()

    if it % log_interval == 0:
        dt = time.time() - t0
        print(f"iter {it}: loss {loss.item():.4f}, {dt*1000/log_interval:.1f} ms/iter")
        t0 = time.time()

print(f"\ntraining done. best val loss: {best_val_loss:.4f}")
print("generate text with: python sample.py")
