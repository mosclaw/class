"""
Piece 4 of a tiny Mixture of Experts language model: looking inside.

Loads the model trained by lm_train.py and lets you watch the routers.

  --prompt TEXT   which experts handle this text? Prints the most used
                  experts per floor, and which experts each character went to.
  --layer N       show only floor N (0 = first, 3 = last).
  --ban 1,4,5     send these experts home: the router may not pick them.
                  Applies to --layer if given, otherwise to every floor.
  --generate N    let the model continue the prompt for N characters.
  --topics        instead of one prompt, measure on lots of real code and
                  poems which share of each topic every expert gets.

Every run also prints how well the model predicts real code and real poems
(lower = better), so a ban can be judged by numbers, not only by eye.

Examples:
  python lm_probe.py --prompt "def add(a, b):" --generate 150
  python lm_probe.py --prompt "O the sea" --layer 3
  python lm_probe.py --topics
  python lm_probe.py --ban 2,5 --layer 3 --prompt "def add(a, b):" --generate 150
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from lm_data import TOPICS, CharTokenizer, Dataset
from lm_model import Config, TinyMoELM

HERE = Path(__file__).parent
CHECKPOINT = HERE / "checkpoints" / "tiny_moe.pt"
SCORE_BATCHES = 20
BATCH = 32


def load(path):
    ckpt = torch.load(path, weights_only=True)
    cfg = Config(**ckpt["config"])
    model = TinyMoELM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer("")
    tok.chars = ckpt["chars"]
    tok.stoi = {c: i for i, c in enumerate(tok.chars)}
    tok.unknown = tok.stoi.get("?", 0)
    return model, tok


def apply_bans(model, ban, layer):
    floors = [layer] if layer is not None else range(model.cfg.n_layers)
    for f in floors:
        model.ban(f, ban)


@torch.no_grad()
def score(model, ds):
    """Average loss per topic on held-out text the model never trained on."""
    g = torch.Generator().manual_seed(1234)
    sums, counts = torch.zeros(len(TOPICS)), torch.zeros(len(TOPICS))
    for _ in range(SCORE_BATCHES):
        x, y, topic = ds.get_batch("val", BATCH, model.cfg.block_size, generator=g)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none")
        sums.index_add_(0, topic.view(-1), loss)
        counts.index_add_(0, topic.view(-1), torch.ones_like(loss))
    return dict(zip(TOPICS, (sums / counts).tolist()))


@torch.no_grad()
def topic_shares(model, ds):
    """For each floor and expert: share of code characters and of poem
    characters it handled (each topic's column adds up to 100%)."""
    E = model.cfg.n_experts
    g = torch.Generator().manual_seed(1234)
    counts = torch.zeros(model.cfg.n_layers, len(TOPICS), E)
    for _ in range(SCORE_BATCHES):
        x, _, topic = ds.get_batch("val", BATCH, model.cfg.block_size, generator=g)
        model(x)
        t = topic.unsqueeze(-1).expand(-1, -1, model.cfg.top_k).reshape(-1)
        for f, choice in enumerate(model.routing()):
            counts[f].index_put_((t, choice.reshape(-1)), torch.ones(t.numel()), accumulate=True)
    return counts / counts.sum(dim=2, keepdim=True)


def print_prompt_routing(model, tok, prompt, floors):
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        model(ids)
    text = tok.decode(ids[0].tolist())
    for f in floors:
        choice = model.routing()[f][0]                     # [T, top_k]
        counts = torch.bincount(choice.flatten(), minlength=model.cfg.n_experts)
        share = counts.float() / counts.sum()
        order = share.argsort(descending=True)
        top = ", ".join(f"{e}: {100 * share[e]:.0f}%" for e in order.tolist() if counts[e] > 0)
        print(f"\nfloor {f}  most used experts  {top}")
        print("  char  experts")
        for ch, pair in zip(text, choice.tolist()):
            shown = {"\n": "\\n", " ": "' '"}.get(ch, ch)
            print(f"  {shown:>4}  {'+'.join(map(str, pair))}")


def print_topic_table(model, ds):
    shares = topic_shares(model, ds)
    print("share of each topic's characters sent to each expert")
    print("(an expert that loves code and ignores poems is a code specialist)")
    for f in range(model.cfg.n_layers):
        print(f"\nfloor {f}   " + "".join(f"  E{e:<3}" for e in range(model.cfg.n_experts)))
        for t, name in enumerate(TOPICS):
            print(f"  {name:<6}  " + "".join(f"{100 * s:>4.0f}% " for s in shares[f, t].tolist()))
        # How different are the two rows? 0% = identical, 100% = no overlap.
        diff = 0.5 * (shares[f, 0] - shares[f, 1]).abs().sum().item()
        print(f"  how differently code and poems are routed: {100 * diff:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Look inside the tiny MoE model.")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--ban", type=str, default="")
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--topics", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    model, tok = load(args.checkpoint)
    ban = [int(e) for e in args.ban.split(",") if e.strip()]
    if ban:
        apply_bans(model, ban, args.layer)
        where = f"floor {args.layer}" if args.layer is not None else "every floor"
        print(f"banned experts {ban} on {where}")

    ds = Dataset()
    scores = score(model, ds)
    print("how well it predicts real text (loss, lower = better)")
    for name, value in scores.items():
        print(f"  {name:<5} {value:.3f}")

    if args.topics:
        print()
        print_topic_table(model, ds)

    if args.prompt:
        floors = [args.layer] if args.layer is not None else range(model.cfg.n_layers)
        print_prompt_routing(model, tok, args.prompt, floors)

    if args.prompt and args.generate:
        g = torch.Generator().manual_seed(0)
        idx = torch.tensor([tok.encode(args.prompt)])
        out = model.generate(idx, args.generate, temperature=0.7, generator=g)
        print(f"\n--- the model continues ---\n{tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
