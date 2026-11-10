"""
Piece 2 of a tiny Mixture of Experts language model: the model.

A normal transformer layer has two parts:
    attention     - each character looks back at earlier characters
    feed-forward  - a small MLP that processes each character on its own

In an MoE transformer the feed-forward part is replaced by several MLPs
("experts") and a router (a tiny classifier, just one linear layer):

    character ──► router ──► scores for expert 0..7
                         └─► keep the top 2, run only those two experts,
                             blend their outputs by the router's weights

Every layer has its own router and its own experts, so layer 0 can split
characters by something simple (letter vs space vs bracket) while a later
layer can split them by what they mean.

Two hooks are built in for the experiments later:
    model.ban(layer, [3, 5])  - the router may never pick these experts
    layer.moe.last_choice     - which experts each character was sent to

Run:  python lm_model.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 96
    block_size: int = 128    # how many characters the model can look back at
    d_model: int = 128       # size of each character's vector
    n_layers: int = 4
    n_heads: int = 4
    n_experts: int = 8
    top_k: int = 2           # experts that run per character
    expert_hidden: int = 256
    balance_weight: float = 0.01


# ----------------------------------------------------------------------------
# Attention: unchanged from a normal transformer
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # no peeking ahead
        return self.out(y.transpose(1, 2).reshape(B, T, D))


# ----------------------------------------------------------------------------
# The MoE layer: router + experts
# ----------------------------------------------------------------------------
class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.expert_hidden), nn.GELU(),
                nn.Linear(cfg.expert_hidden, cfg.d_model),
            )
            for _ in range(cfg.n_experts)
        )
        self.register_buffer("banned", torch.zeros(cfg.n_experts, dtype=torch.bool))
        self.last_choice = None   # [B, T, top_k] expert ids, kept for probing
        self.balance_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)                                   # [N, D], N = B*T

        logits = self.router(flat)                                # [N, E]
        logits = logits.masked_fill(self.banned, float("-inf"))   # banned experts never win
        top_logits, top_ids = logits.topk(self.top_k, dim=1)      # [N, k]
        weights = F.softmax(top_logits, dim=1)                    # blend of the chosen k

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (top_ids == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:                            # unused experts do no work
                continue
            out.index_add_(0, token_idx, expert(flat[token_idx]) * weights[token_idx, slot, None])

        # Balance loss (from the Switch Transformer paper): smallest when every
        # expert gets the same share of characters. Without it the router tends
        # to send almost everything to one or two experts.
        probs = F.softmax(logits, dim=1)
        share = torch.zeros(self.n_experts, device=x.device)
        share.index_add_(0, top_ids.reshape(-1), torch.ones(top_ids.numel(), device=x.device))
        share = share / top_ids.numel()
        self.balance_loss = self.n_experts * (share * probs.mean(dim=0)).sum()

        self.last_choice = top_ids.view(B, T, self.top_k).detach()
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


# ----------------------------------------------------------------------------
# The whole language model
# ----------------------------------------------------------------------------
class TinyMoELM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        task = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        balance = sum(layer.moe.balance_loss for layer in self.layers) / len(self.layers)
        return logits, task + self.cfg.balance_weight * balance

    def ban(self, layer, experts):
        """Forbid the router of one layer from picking these experts."""
        banned = self.layers[layer].moe.banned
        banned.zero_()
        banned[list(experts)] = True

    def routing(self):
        """Expert choices of the last forward pass: one [B, T, top_k] per layer."""
        return [layer.moe.last_choice for layer in self.layers]

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, generator=None):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        one_expert = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters())
        all_experts = one_expert * self.cfg.n_experts * self.cfg.n_layers
        active = total - all_experts + one_expert * self.cfg.top_k * self.cfg.n_layers
        return total, active


# ----------------------------------------------------------------------------
# Look at what we built (untrained, so the routing is still random)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    model = TinyMoELM(cfg)
    total, active = model.param_counts()
    print(f"{cfg.n_layers} layers x {cfg.n_experts} experts, {cfg.top_k} run per character")
    print(f"weights: {total:,} total, {active:,} run per character ({100 * active / total:.0f}%)")

    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    print(f"\ninput {tuple(idx.shape)} -> logits {tuple(logits.shape)}, loss {loss.item():.2f}")
    print(f"(a random guess over {cfg.vocab_size} characters would score ln({cfg.vocab_size}) = "
          f"{torch.log(torch.tensor(float(cfg.vocab_size))).item():.2f})")

    print("\nexperts picked for the first 8 characters, per layer")
    for i, choice in enumerate(model.routing()):
        print(f"  layer {i}: " + "  ".join("+".join(map(str, pair)) for pair in choice[0, :8].tolist()))

    model.ban(0, [0, 1, 2])
    model(idx)
    used = set(model.routing()[0].flatten().tolist())
    print(f"\nafter ban(0, [0, 1, 2]) layer 0 uses experts {sorted(used)}")
