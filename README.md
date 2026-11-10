# Tiny Mixture of Experts

A learning project: a small router network looks at each input and picks which
one of four expert networks to run, so only about a quarter of the model's
weights are used per input.

## Run

```bash
pip install torch
```

```bash
python3 moe.py
```

On Linux without a GPU, the CPU-only build is much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Training takes about 10 seconds on a 4-core CPU.

### Windows

Install Python 3.10–3.13 from [python.org](https://www.python.org/downloads/)
(the `python` and `python3` commands that come with Windows only open the
Microsoft Store). Then, in PowerShell from this folder:

```powershell
py -3.12 -m venv .venv
```

```powershell
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.venv\Scripts\python.exe moe.py
```

If you have [uv](https://docs.astral.sh/uv/), it can fetch Python for you
instead of the python.org installer:

```powershell
uv venv --python 3.12 .venv
```

```powershell
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
```

Then run `.venv\Scripts\python.exe moe.py` as above. A warning that NumPy
failed to initialize is harmless; the script doesn't use NumPy.

## Expected output

```
model                              test acc   weights run per input
dense MLP                             71.1%            7,013 (100%)
MoE, soft (all experts run)           69.3%            7,128 (100%)
MoE, hard (router picks 1 expert)     69.3%            1,833 (26%)

Where the router sent each hidden domain (rows: domain, columns: expert)
        expert0  expert1  expert2  expert3
domain0       0        0     1254        0
domain1       0        0        0     1240
domain2    1269        0        0        0
domain3       0     1237        0        0
```

The router was never told which domain an input came from; it worked out the
split on its own. Hard routing matches soft routing while running 26% of the
weights, and lands about 2 points below a dense network of the same total size.

## Things to try

- Set `BALANCE_WEIGHT = 0` and watch the router collapse onto fewer experts.
- Set `SHARPNESS_WEIGHT = 0` and compare soft vs hard accuracy.
- Change `N_EXPERTS` to 2 or 8 and read the routing table again.
- Move the domain centres closer together (the `* 3.0` in the data section) so
  the router's job gets harder.
