# Running Bubbleformer on a Lambda Cloud GPU

> ⚠️ **Bubbleformer is not my work.** The model, dataset, and original code were created by
> **HPCForge** (NeurIPS 2025), MIT licensed —
> [paper](https://neurips.cc/virtual/2025/loc/san-diego/poster/121854) ·
> [weights](https://huggingface.co/hpcforge/Bubbleformer) ·
> [dataset](https://huggingface.co/datasets/hpcforge/BubbleML_2). This document is just a
> setup/troubleshooting guide I wrote for running their model.

Step-by-step guide for running Bubbleformer inference on a Lambda Cloud GPU instance
(Linux + CUDA) over SSH. Tested end-to-end on a self-generated `PoolBoiling` dataset with the
`Bubbleformer-S-PB-Saturated` checkpoint.

> **Why Lambda is easier than a Mac:** On a Lambda GPU box you're on Linux + CUDA, so the
> `.cuda()` calls and the `pytorch-cu124` build in `pyproject.toml` work as intended — **no CPU
> edits needed.**

> **Heads-up — this repo copy already contains the fixes.** Several bugs in the stock repo have
> already been patched here (see [What was changed](#appendix-a--what-was-changed-from-the-stock-repo)).
> If you copy *this* folder up, you get them for free. If you `git clone` the original repo instead,
> you'll have to re-apply them.

---

## Fill these in first

Replace these throughout the guide with your own values:

| Placeholder | Example | What it is |
|-------------|---------|------------|
| `KEY`       | `/Users/you/Downloads/BubbleID.pem` | Path to your Lambda SSH key (on your Mac) |
| `IP`        | `163.192.15.22` | Your instance's IP (from the Lambda dashboard) |
| `DATA.hdf5` | `PoolBoiling-Twall-150.hdf5` | Your simulation file |
| `CKPT`      | `Bubbleformer-S-PB-Saturated.ckpt` | The checkpoint that matches your data |

> **Tip on commands:** when a command is shown split across lines with `\` at the end, you can always
> paste it as **one long line instead**. Stray `\` characters mid-line (from a bad copy-paste) are a
> common cause of errors like `mkdir ~/...: No such file or directory`.

---

## Step 0 — Get the code and data onto the instance

Run these **from your Mac** (a local terminal, not an SSH session).

```bash
# 1. create the project + data dir on the instance first (scp won't create parent dirs)
ssh -i KEY ubuntu@IP "mkdir -p ~/Bubbleformer-main/data"

# 2. copy the project folder up -> lands at /home/ubuntu/Bubbleformer-main
scp -i KEY -r /Users/you/Downloads/Bubbleformer-main ubuntu@IP:/home/ubuntu/

# 3. copy your dataset file into the project's data/ dir
scp -i KEY /Users/you/Downloads/DATA.hdf5 ubuntu@IP:/home/ubuntu/Bubbleformer-main/data/
```

> **Bug we hit:** `scp: dest open ".../data/": Failure` means the `data/` folder didn't exist yet.
> `scp` does **not** create parent directories — that's why step 1 (`mkdir -p`) comes first.

Now SSH in and confirm the GPU is visible:

```bash
ssh -i KEY ubuntu@IP
nvidia-smi
```

## Step 1 — Install `uv`

Lambda images don't ship `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc      # or restart shell
uv --version
```

## Step 2 — Create env + install

```bash
cd ~/Bubbleformer-main   # project root
uv venv
source .venv/bin/activate
uv sync                  # installs torch 2.5.0 + cu124 on Linux
uv pip install -e .
```

Verify CUDA is wired up:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and your GPU name printed.

> **Bugs we hit here (already fixed in this repo):**
> - *`Multiple top-level packages discovered in a flat-layout`* — setuptools choked because
>   `data/`, `media/`, `samples/` look like packages. Fixed by adding to `pyproject.toml`:
>   ```toml
>   [tool.setuptools.packages.find]
>   include = ["bubbleformer*"]
>   ```
> - *`ModuleNotFoundError: No module named 'transformers'`* — `transformers` was missing from the
>   dependency list. Fixed by adding `"transformers"` to `pyproject.toml`. (If it still complains,
>   run `uv pip install transformers`.)

## Step 3 — Download the model weights

```bash
uv pip install "huggingface_hub[cli]"

# download ONLY the checkpoint you need (the full repo is large)
huggingface-cli download hpcforge/Bubbleformer CKPT --local-dir ./weights
```

This puts the file at `./weights/CKPT`. Browse the list at
<https://huggingface.co/hpcforge/Bubbleformer/tree/main>.

**Checkpoint naming:** `S`/`L` = Small (232 MB) / Large (923 MB); `PB`/`FB` = Pool / Flow Boiling;
then the regime (`Saturated`, `Subcooled`, `VelScale`, `SingleBubble`).

- The **canonical default** model is **Small** (`default.yaml` uses `film_avit_small`).
- Match the **regime + fluid** to your data. For an FC-72 *saturated pool boiling* sim, use
  `Bubbleformer-S-PB-Saturated.ckpt`.
- `inference.py` reads the architecture *from the checkpoint itself*, so any checkpoint loads —
  but only a matching one gives meaningful predictions.

## Step 4 — Create the fluid-parameters JSON sidecar  ⚠️ easy to miss

The `Bubbleformer-S/L` models are **FiLM-AViT**, which condition on fluid parameters. The dataset
loads them from a **`.json` file sitting next to your `.hdf5`**, with the same base name:
`data/DATA.json`. **Without it you get `FileNotFoundError: ... .json`.**

> **Why you need to make this yourself:** if you generated your own simulation, there's no sidecar
> to download — you must create it from your simulation's parameters. (HF dataset files ship with
> their own sidecars.)

Create `data/DATA.json` with these **9 keys** (values below are the FC-72 saturated example —
replace with *your* simulation's nondimensionalized values; `wallTemp` comes from your case):

```json
{
  "inv_reynolds": 0.0043,
  "cpgas": 0.7997,
  "mugas": 0.02816,
  "rhogas": 0.008687,
  "thcogas": 0.209,
  "stefan": 1.2221,
  "prandtl": 7.35,
  "heater": {
    "nucWaitTime": 0.4,
    "wallTemp": 150
  }
}
```

| Key | Meaning |
|-----|---------|
| `inv_reynolds` | 1 / Reynolds number |
| `cpgas` | specific heat of the vapor phase |
| `mugas` | dynamic viscosity of the vapor phase |
| `rhogas` | density of the vapor phase |
| `thcogas` | thermal conductivity of the vapor phase |
| `stefan` | Stefan number |
| `prandtl` | Prandtl number |
| `heater.nucWaitTime` | nucleation wait time at the heater |
| `heater.wallTemp` | wall temperature (e.g. 150 for a `Twall-150` case) |

All 9 must be present or you'll get a `KeyError`. Make the file on your Mac and `scp` it up, or
create it directly on the instance, ending at `./data/DATA.json`.

## Step 5 — Point the inference script at your files

`scripts/inference.py` ships pointed at the authors' HPC cluster paths and a `scot` checkpoint.
In **this repo it's already edited** for the example run; for your own files set:

- **`weights_path`** (~line 174) → `"./weights/CKPT"`
- **`test_path`** (~line 184) → `["./data/DATA.hdf5"]`
- **`save_dir`** (~line 235) → `"./outputs/run1"`

Leave the `.cuda()` lines alone — you have a GPU.

> **Edit locally, then `scp` the file up** — don't hand-edit with `nano` on the instance *and*
> `scp` from your Mac, or one will clobber the other (we hit exactly this). Pick the local file as
> the single source of truth:
> ```bash
> scp -i KEY /Users/you/Downloads/Bubbleformer-main/scripts/inference.py ubuntu@IP:/home/ubuntu/Bubbleformer-main/scripts/inference.py
> ```

### FiLM-AViT fixes already applied to `inference.py`

The stock script was written for `scot` (no fluid params) and a fixed 200-step rollout. For the
FiLM-AViT checkpoints it needed these changes (already in this repo):

```python
# dataset: turn fluid params ON
return_fluid_params=True,

# the autoregressive loop:
for itr in range(0, len(test_dataset), skip_itrs):       # was range(0, 200, ...)
    inp, tgt, fluid_params = test_dataset[itr]           # was: inp, tgt = ...
    ...
    inp = inp.cuda().float().unsqueeze(0)
    fluid_params = fluid_params.cuda().float().unsqueeze(0)
    pred = model(inp, fluid_params)                      # was: model(inp)
```

> **Bugs these fix:**
> - `TypeError: FiLMConditionedAViT.forward() missing 1 required positional argument: 'fluid_params'`
>   → the model needs the fluid-param vector passed in.
> - `IndexError: list index out of range` near the end of the rollout → the loop was hardcoded to
>   200 steps; `len(test_dataset)` stops it at however many timesteps your data actually has.

## Step 6 — Run inference

```bash
python scripts/inference.py
```

You'll see per-step lines like `Autoreg pred 0, inp tw [100, 105]...` followed by a loss value,
repeating until the data runs out. Then it writes to `./outputs/run1/`:

- `predictions.pt` — raw prediction/target tensors (saved **before** plotting).
- `relative_l2_error.png` — error-over-time curve.
- `plots/0000.png, 0001.png, …` — per-timestep figures (top row = ground truth, bottom = prediction).

> **"0 files done" and a long pause is normal.** That's the **plotting** stage (CPU matplotlib with
> slow `streamplot` velocity streamlines), not a hang. It prints progress every 25 frames. Watch it
> progress from a second SSH session: `watch -n 2 'ls outputs/run1/plots/ | wc -l'`.

## Step 7 — View the results (pull them to your Mac)

Run **on your Mac**, as **one line**, full paths (no `~`, no stray `\`):

```bash
scp -i KEY -r ubuntu@IP:/home/ubuntu/Bubbleformer-main/outputs/run1 /Users/you/Downloads/
open /Users/you/Downloads/run1
```

Start with `plots/0000.png` (label vs. prediction side by side).

> **Bug we hit:** `scp: mkdir ~/Downloads/...: No such file or directory` — caused by a backslash
> stuck to `~` (`\~`) when a multi-line command was pasted and the line breaks collapsed, which stops
> the shell from expanding `~`. Fix: one single line, use the **full path** `/Users/you/Downloads/`,
> and make sure you're on your **Mac**, not the instance.

---

## Reading the output (sanity check)

The relative-L2 loss grows over the rollout. Some growth is normal (autoregressive error compounds),
but **very large values (hundreds)** signal a **distribution mismatch** between your data and the
training set. Most likely causes, in order:

1. **Normalization** — the script uses `norm="none"` and pulls normalization constants from the
   checkpoint. If your data's physical ranges differ (the plotting code assumes temperature ∈
   [58, 92]), the model sees out-of-distribution inputs.
2. **Fluid-param scaling** — if your 9 JSON values aren't nondimensionalized the same way as
   training, the FiLM conditioning is off.
3. **Field order / units** — channels must be `[dfun, temperature, velx, vely]`, with `dfun` a
   signed distance function.

It will still run and produce plots regardless; matching preprocessing is what improves accuracy.

---

## Optional — run the Jupyter notebook instead

`scripts/inference_autoregressive.ipynb` does the same thing interactively, but **still has the
original cluster paths and the SCOT/FiLM mismatch** — it won't run top-to-bottom without the same
edits as Step 5. To reach it from your Mac browser you need **two terminals + the browser**:

```bash
# Terminal 1 (Mac, SSH'd into instance): start the server with a known token
cd ~/Bubbleformer-main && source .venv/bin/activate
jupyter notebook --no-browser --port 8888 --IdentityProvider.token=bubble

# Terminal 2 (Mac, local): open the tunnel and leave it running
ssh -i KEY -L 8888:localhost:8888 ubuntu@IP
```

Then open in your Mac browser: `http://127.0.0.1:8888/tree?token=bubble`

> - Setting `--IdentityProvider.token=bubble` saves you hunting for the long auto-generated token.
> - `bind [127.0.0.1]:8888: Address already in use` → a leftover tunnel holds the local port. Either
>   `pkill -f "8888:localhost:8888"` on your Mac, or use a different local port:
>   `-L 8889:localhost:8888` and browse `http://127.0.0.1:8889/...`.

For just viewing results, you don't need Jupyter at all — Step 7's `scp` is simpler.

---

## Optional — training

```bash
# edit bubbleformer/config/default.yaml -> set log_dir, set use_wandb: False (or `wandb login`)
# edit the active data_cfg (e.g. data_cfg/singlebubble.yaml) -> train_paths / val_paths
python scripts/train.py nodes=1 devices=1 max_epochs=400 batch_size=8 expt=forecast
```

---

## Lambda-specific tips

- **Run long jobs inside `tmux`** (`tmux new -s bf`) so a dropped SSH connection doesn't kill them.
- **Keep data on the instance's local NVMe** for fast `.hdf5` reads.
- If `uv sync` complains about the CUDA index, confirm with
  `python -c "import torch; print(torch.version.cuda)"` — it should report `12.4`.

---

## Appendix A — What was changed from the stock repo

These fixes are already applied in this copy. If a groupmate clones the original repo, re-apply them.

1. **`pyproject.toml`** — added explicit package discovery so the editable install builds:
   ```toml
   [tool.setuptools.packages.find]
   include = ["bubbleformer*"]
   ```
2. **`pyproject.toml`** — added the missing `"transformers"` dependency.
3. **`scripts/inference.py`** — repointed `weights_path` / `test_path` / `save_dir` to local
   `./weights`, `./data`, `./outputs`; set `return_fluid_params=True`; unpacked and passed
   `fluid_params` into `model(inp, fluid_params)`; changed the rollout loop to
   `range(0, len(test_dataset), skip_itrs)`.
4. **`data/DATA.json`** — created the fluid-parameter sidecar (not in the repo; per-dataset).

## Appendix B — Faster copy with `rsync`

`scp -r` copies **everything**, including `.venv` and caches. To skip that, use `rsync` from your
Mac:

```bash
rsync -avz --progress \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
  -e "ssh -i KEY" \
  /Users/you/Downloads/Bubbleformer-main/ \
  ubuntu@IP:/home/ubuntu/Bubbleformer-main/
```

> **Note the trailing slashes** on both paths — `Bubbleformer-main/` copies the *contents* into
> `Bubbleformer-main/` (otherwise you'd get a nested `Bubbleformer-main/Bubbleformer-main`).

## Appendix C — Error quick-reference

| Error | Cause | Fix |
|-------|-------|-----|
| `scp: dest open ".../data/": Failure` | parent dir doesn't exist | `mkdir -p ~/Bubbleformer-main/data` first |
| `Multiple top-level packages discovered` | setuptools auto-discovery | `[tool.setuptools.packages.find]` include `bubbleformer*` |
| `No module named 'transformers'` | missing dependency | `uv pip install transformers` |
| `No module named 'torch'` | venv not activated | `source .venv/bin/activate` |
| `FileNotFoundError: ....ckpt` | weights not downloaded / wrong path | download checkpoint; fix `weights_path` |
| `FileNotFoundError: ....json` | missing fluid-param sidecar | create `data/DATA.json` (Step 4) |
| `forward() missing 1 required positional argument: 'fluid_params'` | FiLM model needs fluid params | pass `model(inp, fluid_params)` (Step 5) |
| `IndexError: list index out of range` | rollout longer than data | loop `range(0, len(test_dataset), skip_itrs)` |
| `bind [127.0.0.1]:8888: Address already in use` | local port taken | `pkill -f "8888:localhost:8888"` or use port 8889 |
| `scp: mkdir ~/...: No such file or directory` | stray `\~` / wrong machine | one line, full path, run on your Mac |
