"""

A GPT-style (decoder-only) Transformer language model written from scratch.

This is intentionally heavily commented for learning. Every component maps to a
concept from the "Attention Is All You Need" paper and the GPT papers:

    Input tokens -> Token embeddings + Positional embeddings
                 -> N x Transformer Block (Self-Attention + MLP)
                 -> Final LayerNorm
                 -> Linear head -> logits over the vocabulary

We only build the DECODER half (no cross-attention), because for a language model
we just predict the next token given previous tokens. This is exactly how GPT-2,
Llama, etc. work at the core.

Author: <your name> -- FYP, from-scratch LLM.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """All the knobs that define the model's size and shape.

    The defaults here are SMALL on purpose so you can test on a laptop CPU.
    When you move to the RTX 5090 on the HPC, scale these up (see README).
    """
    block_size: int = 256      # max context length (how many tokens the model sees at once)
    vocab_size: int = 65       # number of distinct tokens (set by your tokenizer/data)
    n_layer: int = 6           # number of Transformer blocks stacked on top of each other
    n_head: int = 6            # number of attention heads (n_embd must be divisible by this)
    n_embd: int = 384          # embedding dimension (the "width" of the model)
    dropout: float = 0.0       # dropout for regularization (0 is fine for small data)
    bias: bool = True          # use bias terms in Linear/LayerNorm? GPT-2 uses True.


# -----------------------------------------------------------------------------
# Building block 1: Causal Self-Attention
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    "Self-attention" = every token looks at other tokens to decide what's relevant.
    "Causal" = a token may only look at itself and tokens BEFORE it (never the future),
    because at generation time we don't know future tokens yet.

    Shapes use: B = batch size, T = sequence length (tokens), C = embedding dim (n_embd).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # One big linear layer that produces query, key, and value for ALL heads at once.
        # Output is 3 * n_embd wide; we split it into q, k, v below. (Efficiency trick.)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: mixes the heads' results back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # PyTorch >= 2.0 has a fast fused attention kernel ("Flash Attention").
        # We use it if available; otherwise fall back to the manual implementation
        # so the code still runs on older versions / CPU.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # The causal mask: a lower-triangular matrix of 1s. Position (i, j) is 1
            # if token i is allowed to attend to token j (i.e. j <= i).
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Project input into query, key, value and split into the three pieces.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape so each head gets its own slice of the embedding dimension.
        # (B, T, C) -> (B, n_head, T, head_size). head_size = C / n_head.
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            # Fused, memory-efficient attention. is_causal=True applies the mask for us.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention so you can SEE the math:
            # 1) scores = q @ k^T / sqrt(head_size)  -> how much each token attends to others
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
            # 2) mask out the future (set to -inf so softmax makes them ~0)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            # 3) softmax turns scores into a probability distribution over past tokens
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # 4) weighted sum of the values
            y = att @ v

        # Reassemble all heads back into a single (B, T, C) tensor.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection + dropout.
        y = self.resid_dropout(self.c_proj(y))
        return y


# -----------------------------------------------------------------------------
# Building block 2: MLP (a.k.a. feed-forward network)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """A simple 2-layer fully-connected network applied to each token independently.

    Attention moves information BETWEEN tokens; the MLP does per-token "thinking".
    The hidden layer is 4x wider than the embedding (a standard ratio from GPT-2).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()  # smooth activation function used by GPT-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
# Building block 3: a Transformer Block
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """One Transformer block = Attention + MLP, each with a LayerNorm and a residual.

    The pattern is "pre-norm": LayerNorm BEFORE the sub-layer, then add the input back
    (the residual connection `x = x + ...`). Residuals let gradients flow through deep
    stacks of blocks without vanishing.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # communicate between tokens
        x = x + self.mlp(self.ln_2(x))   # think per-token
        return x


# -----------------------------------------------------------------------------
# The full GPT model
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte = "word token embedding": maps each token id -> a vector.
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = "word position embedding": maps each position 0..block_size-1 -> a vector.
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            # the stack of Transformer blocks
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer norm before the output head
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # The language-model head: projects the final embedding to a score for every
        # token in the vocabulary. These scores are called "logits".
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and output head share the same weights.
        # This saves parameters and tends to improve quality (a GPT-2 trick).
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights.
        self.apply(self._init_weights)
        # Special scaled init for the residual projections (from the GPT-2 paper).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Count parameters. By default we exclude the position embeddings to match
        how GPT-2 reports its size."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) tensor of token ids   -- the input context
        targets: (B, T) tensor of token ids   -- the "correct next token" at each position
                 (only provided during training; None during generation)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # positions 0..T-1

        tok_emb = self.transformer.wte(idx)   # (B, T, C) token meanings
        pos_emb = self.transformer.wpe(pos)   # (T, C)    position info
        x = self.transformer.drop(tok_emb + pos_emb)  # combine: "what" + "where"

        for block in self.transformer.h:      # run through all Transformer blocks
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # TRAINING: compute logits for every position and the cross-entropy loss.
            logits = self.lm_head(x)
            # Flatten (B, T, vocab) -> (B*T, vocab) and compare to flattened targets.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # INFERENCE: we only need the logits for the LAST position to predict the next token.
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Build an AdamW optimizer. Weights in matmuls get weight decay; biases and
        LayerNorm params do not (a common, well-tested convention)."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Use the fused AdamW kernel on CUDA when available (faster).
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__doc__)
        extra = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation: repeatedly predict the next token and append it.

        idx: (B, T) starting context of token ids.
        temperature: <1.0 = more confident/repetitive, >1.0 = more random/creative.
        top_k: if set, only sample from the k most likely tokens (reduces gibberish).
        """
        for _ in range(max_new_tokens):
            # If the context grew longer than block_size, crop to the last block_size tokens.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)               # get predictions
            logits = logits[:, -1, :] / temperature  # focus on the last step, apply temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")  # mask out everything below top-k
            probs = F.softmax(logits, dim=-1)        # turn logits into probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample one token
            idx = torch.cat((idx, idx_next), dim=1)  # append and continue
        return idx
