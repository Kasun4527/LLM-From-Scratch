# LLM From Scratch

A decoder-only Transformer (GPT-style) language model implemented from scratch in
PyTorch, for understanding how LLMs actually work. Built as the "fundamentals"
component of an FYP on adaptive learning AI.

> **Scope note.** This trains a *small* language model to demonstrate the
> architecture and training process. It is **not** meant to be the production brain
> of the tutoring app — that job belongs to RAG + a production-grade model. A model
> you can train on one GPU will be far weaker than an off-the-shelf LLM. The value
> here is *understanding*, plus a real training run you can write up and defend.

## Files

| File | What it does |
|------|--------------|
| `model.py` | The Transformer itself: embeddings, causal self-attention, MLP, blocks, GPT class, text generation. Heavily commented. |
| `prepare_data.py` | Tokenizes raw text into `data/train.bin`, `data/val.bin`, `data/meta.pkl`. Character-level tokenizer. |
| `train.py` | The training loop: batching, loss, backprop, LR schedule, checkpointing. |
| `sample.py` | Loads a checkpoint and generates text. |

## Quick start (laptop, CPU — sanity check)

This runs in a few minutes on CPU and proves the code is correct *before* you spend
HPC GPU time.

```bash
cd llm-from-scratch
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python prepare_data.py      # downloads tiny-shakespeare, writes data/*.bin
python train.py             # trains a tiny model (~2000 steps)
python sample.py            # generates text from the trained model
```

After ~2000 steps the val loss should drop from ~4.2 to roughly ~1.7–2.0 and the
samples should look like (broken but English-ish) Shakespeare. That's success for
the sanity check — it means the whole pipeline works.

## Training on your own text

```bash
python prepare_data.py --input path/to/your_aiml_notes.txt
python train.py
```

## Scaling up on the HPC (RTX 5090)

The RTX 5090 is Blackwell-architecture, so you need a recent PyTorch + CUDA build.
**Install this first and verify the GPU is visible:**

```bash
# Use the current CUDA wheel (check pytorch.org/get-started for the latest):
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then edit the config block at the top of `train.py` to a real model size, e.g. a
GPT-2-small-class model:

```python
max_iters   = 100000      # or more
block_size  = 1024        # context length
batch_size  = 32          # raise until you nearly fill 32 GB VRAM
n_layer     = 12
n_head      = 12
n_embd      = 768         # -> ~124M parameters (GPT-2 small)
dropout     = 0.0
learning_rate = 6e-4      # lower LR for bigger models
warmup_iters  = 2000
```

Performance tips for the GPU run:
- `bf16` is selected automatically when CUDA + bf16 are available (the 5090 supports it).
- For a real run, switch the data prep to a **BPE tokenizer** (`tiktoken`, vocab 50257)
  and a much larger corpus (e.g. OpenWebText / FineWeb-Edu) — character-level wastes
  capacity learning to spell. See the note at the bottom of `prepare_data.py`.
- `torch.compile(model)` after building the model gives a big speedup on PyTorch 2.x.

## How it works (one paragraph for your report)

Text is split into tokens; each token id is mapped to a learned vector (token
embedding) and added to a learned position vector. The sequence flows through N
Transformer blocks, each doing **causal self-attention** (every token gathers
information from earlier tokens) followed by a per-token **MLP**. A final linear
"head" turns each position's vector into scores (logits) over the whole vocabulary.
During training we compare those predictions to the actual next token using
**cross-entropy loss** and use **backpropagation + AdamW** to adjust the weights.
At generation time we feed the model some context, sample the next token from its
predicted distribution, append it, and repeat.
```
