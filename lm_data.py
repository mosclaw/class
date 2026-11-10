"""
Piece 1 of a tiny Mixture of Experts language model: the data.

A language model learns one thing: given some text, guess the next character.
Before any of that can happen, text has to become numbers. This file:

  1. Collects two kinds of text, so we can later ask "do code and poems use
     different experts?":
       - code:  .py files from Python's own standard library (already on disk)
       - poems: any .txt files you put in data/poems/ (e.g. Leaves of Grass)
  2. Cleans it down to plain ASCII, so the alphabet stays small.
  3. Builds a character tokenizer: every character gets an id, 0..vocab-1.
  4. Splits into train / validation and serves random batches of
     (input chunk, same chunk shifted one character = the answers).

The topic of every character is kept alongside the text, like the hidden
"domain" in moe.py. The model never sees it; we only use it afterwards to
check where the router sent code vs poems.

Run:  python lm_data.py
"""

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
POEM_DIR = HERE / "data" / "poems"
STDLIB_DIR = Path(os.__file__).parent   # where Python's own .py files live

TOPICS = ["code", "poem"]
MAX_CHARS_PER_TOPIC = 1_000_000          # keep the two topics roughly equal
VAL_FRACTION = 0.1


# ----------------------------------------------------------------------------
# 1. Collect text
# ----------------------------------------------------------------------------
def strip_gutenberg(text):
    """Project Gutenberg files wrap the book in a licence header and footer."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[text.index("\n", start) + 1:end]
    return text


def collect(files, limit):
    """Read files in a fixed order until `limit` characters are collected."""
    parts, total = [], 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        parts.append(text)
        total += len(text)
        if total >= limit:
            break
    return "\n\n".join(parts)[:limit]


def load_raw():
    code_files = sorted(p for p in STDLIB_DIR.glob("*.py"))
    poem_files = sorted(POEM_DIR.glob("*.txt")) if POEM_DIR.exists() else []
    poems = "\n\n".join(strip_gutenberg(p.read_text(encoding="utf-8")) for p in poem_files)
    poems = clean(poems)[:MAX_CHARS_PER_TOPIC]
    # Take as much code as there are poems, so neither topic dominates training.
    code = clean(collect(code_files, MAX_CHARS_PER_TOPIC))[:len(poems) or MAX_CHARS_PER_TOPIC]
    return {"code": code, "poem": poems}


# ----------------------------------------------------------------------------
# 2. Clean: everything becomes printable ASCII plus newline
# ----------------------------------------------------------------------------
REPLACE = {
    "\t": "    ", "\r": "",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "--", "…": "...", " ": " ",
}


def clean(text):
    for old, new in REPLACE.items():
        text = text.replace(old, new)
    return "".join(c for c in text if c == "\n" or " " <= c <= "~")


# ----------------------------------------------------------------------------
# 3. Tokenizer: one id per character
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.unknown = self.stoi.get("?", 0)   # unseen characters become "?"

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi.get(c, self.unknown) for c in clean(text)]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


# ----------------------------------------------------------------------------
# 4. Dataset: token ids + the topic of each token, split into train / val
# ----------------------------------------------------------------------------
class Dataset:
    def __init__(self):
        raw = {name: clean(text) for name, text in load_raw().items()}
        self.tokenizer = CharTokenizer("".join(raw.values()))

        splits = {"train": ([], []), "val": ([], [])}
        for topic_id, name in enumerate(TOPICS):
            ids = self.tokenizer.encode(raw[name])
            cut = int(len(ids) * (1 - VAL_FRACTION))
            for split, chunk in (("train", ids[:cut]), ("val", ids[cut:])):
                splits[split][0].extend(chunk)
                splits[split][1].extend([topic_id] * len(chunk))

        self.ids = {s: torch.tensor(t, dtype=torch.long) for s, (t, _) in splits.items()}
        self.topic = {s: torch.tensor(t, dtype=torch.long) for s, (_, t) in splits.items()}
        self.chars_per_topic = {name: len(raw[name]) for name in TOPICS}

    def get_batch(self, split, batch_size, block_size, generator=None):
        """Random chunks of text. y is x shifted by one: the next character."""
        data = self.ids[split]
        start = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
        offsets = torch.arange(block_size)
        x = data[start[:, None] + offsets]
        y = data[start[:, None] + offsets + 1]
        topic = self.topic[split][start[:, None] + offsets]
        return x, y, topic


# ----------------------------------------------------------------------------
# Look at what we built
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ds = Dataset()
    tok = ds.tokenizer

    print("characters per topic")
    for name, n in ds.chars_per_topic.items():
        print(f"  {name:<5} {n:>10,}")
    if ds.chars_per_topic["poem"] == 0:
        print(f"\n  No poems found. Put .txt files in {POEM_DIR}", file=sys.stderr)

    print(f"\nvocabulary: {tok.vocab_size} characters")
    print("  " + repr("".join(tok.chars)))

    sample = "def sea(wave):\n    return 'O sea!'"
    ids = tok.encode(sample)
    print(f"\nencode {sample!r}\n  -> {ids}")
    print(f"decode back -> {tok.decode(ids)!r}")

    g = torch.Generator().manual_seed(0)
    x, y, topic = ds.get_batch("train", batch_size=2, block_size=40, generator=g)
    print("\none training batch (the model reads x and must predict y)")
    for row in range(2):
        print(f"  [{TOPICS[topic[row, 0]]}]")
        print(f"    x: {tok.decode(x[row].tolist())!r}")
        print(f"    y: {tok.decode(y[row].tolist())!r}")
