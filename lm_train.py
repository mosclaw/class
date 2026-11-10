"""
Piece 3 of a tiny Mixture of Experts language model: training.

Training is a loop of one tiny lesson repeated thousands of times:
  1. take random chunks of text            (piece 1: Dataset.get_batch)
  2. let the model guess every next char   (piece 2: TinyMoELM)
  3. measure how wrong it was (the loss) and nudge every weight a little
     in the direction that would have made it less wrong

The routers are trained by the same nudges. Nobody tells a router "this is
code" or "this is a poem"; it only learns which experts make the guesses
better. Piece 4 will check whether that split code and poems apart anyway.

Every few hundred steps the script prints the validation loss separately
for code and poems (lower is better; ln(96) = 4.56 is random guessing), and
at the end it writes a short sample from a code prompt and a poem prompt.

The trained model is saved to checkpoints/tiny_moe.pt for piece 4.

Run:  python lm_train.py              (about 9 minutes on a 6-core CPU)
      python lm_train.py --steps 300  (quick check, about 1 minute)
"""

import argparse
import math
import time
from pathlib import Path

import torch

from lm_data import TOPICS, Dataset
from lm_model import Config, TinyMoELM

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "checkpoints" / "tiny_moe.pt"

BATCH = 32
LR = 2e-3
WARMUP = 100
EVAL_EVERY = 250
EVAL_BATCHES = 20


def learning_rate(step, total):
    """Ramp up for WARMUP steps, then glide down along a cosine curve."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, total - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ds, cfg):
    """Validation loss per topic, on a fixed set of batches."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    sums, counts = torch.zeros(len(TOPICS)), torch.zeros(len(TOPICS))
    for _ in range(EVAL_BATCHES):
        x, y, topic = ds.get_batch("val", BATCH, cfg.block_size, generator=g)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none"
        )
        sums.index_add_(0, topic.view(-1), loss)
        counts.index_add_(0, topic.view(-1), torch.ones_like(loss))
    model.train()
    return (sums / counts.clamp_min(1)).tolist()


def expert_share(model, ds, cfg):
    """Fraction of characters each expert handled, per layer, on val text."""
    g = torch.Generator().manual_seed(99)
    x, _, _ = ds.get_batch("val", BATCH, cfg.block_size, generator=g)
    with torch.no_grad():
        model(x)
    return [
        torch.bincount(choice.flatten(), minlength=cfg.n_experts).float() / choice.numel()
        for choice in model.routing()
    ]


def sample(model, tok, prompt, n_chars=200):
    g = torch.Generator().manual_seed(0)
    idx = torch.tensor([tok.encode(prompt)])
    model.eval()
    out = model.generate(idx, n_chars, temperature=0.7, generator=g)
    model.train()
    return tok.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    torch.manual_seed(0)
    ds = Dataset()
    tok = ds.tokenizer
    cfg = Config(vocab_size=tok.vocab_size)
    model = TinyMoELM(cfg)
    total, active = model.param_counts()
    print(f"training {total:,} weights ({active:,} run per character) for {args.steps} steps")
    print(f"{'step':>6}{'code loss':>11}{'poem loss':>11}{'elapsed':>10}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    start = time.time()
    for step in range(args.steps + 1):
        if step % EVAL_EVERY == 0 or step == args.steps:
            code_loss, poem_loss = evaluate(model, ds, cfg)
            print(f"{step:>6}{code_loss:>11.3f}{poem_loss:>11.3f}{time.time() - start:>9.0f}s", flush=True)
        if step == args.steps:
            break

        for group in opt.param_groups:
            group["lr"] = learning_rate(step, args.steps)
        x, y, _ = ds.get_batch("train", BATCH, cfg.block_size)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict(), "chars": tok.chars}, args.out)
    print(f"\nsaved {args.out}")

    print("\nshare of characters each expert handled (balance check, 12% = perfectly even)")
    for i, share in enumerate(expert_share(model, ds, cfg)):
        print(f"  layer {i}: " + " ".join(f"{100 * s:>3.0f}%" for s in share.tolist()))

    for prompt in ("def read_lines(path):\n", "O the sea, "):
        print(f"\n--- sample from {prompt!r} ---")
        print(sample(model, tok, prompt))


if __name__ == "__main__":
    main()
