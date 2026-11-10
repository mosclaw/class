"""
A tiny Mixture of Experts (MoE), built to learn how a small "router" network
can pick which part of a bigger model to run for each input.

    input x ──► router (tiny classifier) ──► "use expert 2"
            └─► expert 2 (the only weights that run) ──► prediction

Run:  python3 moe.py

What the script does
  1. Makes a synthetic dataset with 4 hidden "domains". Each domain has its own
     labelling rule, so a model has to behave differently per domain.
     The domain id is never shown to any model.
  2. Trains a dense baseline: one MLP, all weights used on every input.
  3. Trains an MoE: 4 small experts + a router. Training is "soft" (every
     expert runs and the router's probabilities weight them) so gradients
     reach the router.
  4. Evaluates the MoE "hard": the router picks ONE expert per input and only
     that expert's weights are executed.
  5. Prints which expert each hidden domain was sent to, so you can see that
     the router discovered the domains on its own.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DIM = 16          # input features
N_DOMAINS = 4     # hidden domains in the data
N_CLASSES = 5
N_EXPERTS = 4     # try 2 or 8 and see what the routing table looks like
EXPERT_HIDDEN = 32
EPOCHS = 30
BATCH = 256
LR = 3e-3

# Weights of the two extra router losses (set either to 0 to see why it exists)
BALANCE_WEIGHT = 0.5    # push the router to use all experts overall
SHARPNESS_WEIGHT = 0.1  # push the router to be confident on each single input


# ----------------------------------------------------------------------------
# 1. Data: each domain is a cluster in input space with its own random
#    "teacher" network that decides the label.
# ----------------------------------------------------------------------------
def make_data(n, generator, centers, teachers):
    domain = torch.randint(0, N_DOMAINS, (n,), generator=generator)
    local = torch.randn(n, DIM, generator=generator)   # position inside cluster
    x = centers[domain] + local
    y = torch.empty(n, dtype=torch.long)
    for d in range(N_DOMAINS):
        mask = domain == d
        w1, w2 = teachers[d]
        y[mask] = (torch.tanh(local[mask] @ w1) @ w2).argmax(dim=1)
    return x, y, domain


g = torch.Generator().manual_seed(0)
centers = torch.randn(N_DOMAINS, DIM, generator=g) * 3.0
teachers = [
    (torch.randn(DIM, 24, generator=g), torch.randn(24, N_CLASSES, generator=g))
    for _ in range(N_DOMAINS)
]
x_train, y_train, _ = make_data(20_000, g, centers, teachers)
x_test, y_test, dom_test = make_data(5_000, g, centers, teachers)


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def mlp(hidden):
    return nn.Sequential(
        nn.Linear(DIM, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, N_CLASSES),
    )


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        # The router is deliberately tiny: a single linear layer.
        self.router = nn.Linear(DIM, N_EXPERTS)
        self.experts = nn.ModuleList(mlp(EXPERT_HIDDEN) for _ in range(N_EXPERTS))

    def forward_soft(self, x):
        """Training path: run every expert, blend by router probabilities.

        Returns log p(y|x) where p = sum_i gate_i * p_i(y|x), plus the gates.
        Mixing probabilities (not logits) rewards the router for sending an
        input to the one expert that is good at it, which drives specialisation.
        """
        gates = F.softmax(self.router(x), dim=1)                     # [B, E]
        expert_logp = torch.stack(
            [F.log_softmax(e(x), dim=1) for e in self.experts], dim=1
        )                                                            # [B, E, C]
        logp = torch.logsumexp(gates.clamp_min(1e-9).log().unsqueeze(2) + expert_logp, dim=1)
        return logp, gates

    @torch.no_grad()
    def forward_hard(self, x):
        """Inference path: router picks one expert; only that expert runs."""
        choice = self.router(x).argmax(dim=1)                        # [B]
        out = torch.empty(x.shape[0], N_CLASSES)
        for i, expert in enumerate(self.experts):
            mask = choice == i
            if mask.any():                 # experts with no inputs do no work
                out[mask] = expert(x[mask])
        return out, choice


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ----------------------------------------------------------------------------
# 3. Training
# ----------------------------------------------------------------------------
def batches():
    perm = torch.randperm(x_train.shape[0])
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        yield x_train[idx], y_train[idx]


def train_dense(model):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for xb, yb in batches():
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()


def train_moe(model):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    uniform = torch.full((N_EXPERTS,), 1.0 / N_EXPERTS)
    for epoch in range(EPOCHS):
        for xb, yb in batches():
            logp, gates = model.forward_soft(xb)
            task_loss = F.nll_loss(logp, yb)

            # Balance: averaged over the batch, each expert should get an equal
            # share. Without this the router can collapse onto one expert.
            usage = gates.mean(dim=0)
            balance_loss = (usage * (usage / uniform).log()).sum()   # KL(usage || uniform)

            # Sharpness: for one input the router should commit to one expert,
            # so that switching to hard top-1 routing at inference loses nothing.
            entropy = -(gates * gates.clamp_min(1e-9).log()).sum(dim=1).mean()

            loss = task_loss + BALANCE_WEIGHT * balance_loss + SHARPNESS_WEIGHT * entropy
            opt.zero_grad(); loss.backward(); opt.step()


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item() * 100


# ----------------------------------------------------------------------------
# 4. Run the experiment
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    moe = MoE()
    total = n_params(moe)
    one_expert = n_params(moe.experts[0])
    router = n_params(moe.router)

    # Dense baseline sized to roughly the same total parameter count as the MoE.
    dense_hidden = EXPERT_HIDDEN
    while n_params(mlp(dense_hidden + 1)) <= total:
        dense_hidden += 1
    dense = mlp(dense_hidden)

    train_dense(dense)
    train_moe(moe)

    with torch.no_grad():
        dense_acc = accuracy(dense(x_test), y_test)
        soft_logp, _ = moe.forward_soft(x_test)
        soft_acc = accuracy(soft_logp, y_test)
    hard_logits, choice = moe.forward_hard(x_test)
    hard_acc = accuracy(hard_logits, y_test)

    active = router + one_expert
    print(f"{'model':<34}{'test acc':>9}{'weights run per input':>24}")
    print(f"{'dense MLP':<34}{dense_acc:>8.1f}%{n_params(dense):>17,} (100%)")
    print(f"{'MoE, soft (all experts run)':<34}{soft_acc:>8.1f}%{total:>17,} (100%)")
    print(f"{'MoE, hard (router picks 1 expert)':<34}{hard_acc:>8.1f}%{active:>17,} ({100 * active / total:.0f}%)")

    print("\nWhere the router sent each hidden domain (rows: domain, columns: expert)")
    print("        " + "".join(f"expert{i:<3}" for i in range(N_EXPERTS)))
    for d in range(N_DOMAINS):
        counts = torch.bincount(choice[dom_test == d], minlength=N_EXPERTS)
        print(f"domain{d} " + "".join(f"{c:>7}  " for c in counts.tolist()))
