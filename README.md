# COMP 457 — Advanced AI

Coursework for Athabasca COMP 457. Code is shared via git; each machine
builds its own local conda environment (envs contain platform-specific
binaries and are never committed).

## Setup

Prereq: Miniconda or Anaconda installed
(https://docs.conda.io/en/latest/miniconda.html).

Prereq: a conda distribution. Miniforge is recommended because it
defaults to the conda-forge channel this project uses:

- macOS: `brew install --cask miniforge`
- Windows: https://conda-forge.org/download/

Then initialize your shell once and **open a new terminal** (a bare
`conda init` can target the wrong shell, so name it explicitly):

```bash
conda init zsh    # or: conda init bash / conda init powershell
```

### 1. Clone

```bash
git clone https://github.com/BorisBojanov/Comp457_Advanced_Artificial_Intelligence.git
cd Comp457_Advanced_Artificial_Intelligence
```

### 2. Create the conda env from `environment.yml`

```bash
conda env create -f environment.yml
conda activate Comp457
```

If the env already exists and you just want to sync new deps:

```bash
conda env update -f environment.yml --prune
```

### 3. Confirm the right PyTorch build

Step 2 already installed PyTorch — `environment.yml` lists it, and
conda-forge resolves a per-platform build. Do **not** `pip install torch`
into this environment; that combination crashes (see Troubleshooting).

**macOS (Apple Silicon)** — the `cpu_generic_*` build has MPS compiled
in. Nothing further to do.

**Windows with NVIDIA GPU** — conda-forge ships both CPU and CUDA
builds, and the solver only picks a CUDA build if it detects your
driver. Check what you got:

```powershell
conda list pytorch
```

**Windows with NVIDIA GPU** — check your CUDA version with `nvidia-smi`,
then pick the matching index URL. Example for CUDA 12.4:

If the build string starts with `cpu_`, force a CUDA build. Check your
driver's CUDA version with `nvidia-smi`, then match it — conda-forge
currently offers `cuda126`, `cuda128`, and `cuda130`:

```powershell
conda install -c conda-forge "pytorch=2.13.0=cuda126*" torchvision
```

### 4. Verify

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available(), 'mps:', torch.backends.mps.is_available())"
```

Mac should print `mps: True`. Windows should print `cuda: True`.

## Run the Unit 1 quickstart

```bash
python src/unit1_quickstart.py
```

Downloads FashionMNIST to `data/`, trains a small MLP for 5 epochs,
saves weights to `model.pth`, then reloads them for one prediction.

## Troubleshooting

**`CondaError: Run 'conda init' before 'conda activate'`**

`conda activate` needs a shell function that your rc file loads at
startup. Either `conda init` has not run for *this* shell (a bare
`conda init` may modify `.bash_profile` while you actually use zsh), or
the rc file is fine but your terminal tab predates the change. Run
`conda init zsh` and open a new tab — existing tabs keep the old state.

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized`**

Two OpenMP runtimes got loaded into one process. This happens when the
pip PyTorch wheel (which bundles its own `libomp`) is installed next to
conda-forge `numpy` (which links conda's `libomp`). Keep PyTorch on
conda-forge:

```bash
pip uninstall -y torch torchvision
conda install -c conda-forge pytorch=2.13.0 torchvision
```

`KMP_DUPLICATE_LIB_OK=TRUE` only silences the abort; PyTorch's own
warning notes it can produce incorrect results. Don't rely on it.

**`python` runs the wrong interpreter even with the env activated**

A shell alias takes precedence over `PATH`, so a line such as
`alias python=/opt/homebrew/bin/python3` in `~/.zshrc` overrides the
activated environment (and breaks outright if that path no longer
exists). Verify with `type python` — it should print a path under
`envs/Comp457/bin`. Remove any such alias.

**VSCode uses the wrong environment**

Select the interpreter explicitly: Cmd/Ctrl+Shift+P -> *Python: Select
Interpreter* -> the entry under `envs/Comp457/bin/python`. The choice is
cached per workspace, so a stale selection survives env changes and
shows up as `spawn ... ENOENT` errors in the Python extension log.


## Day-to-day workflow between machines

Before starting work on either machine:
'''bash
git pull
'''

After making changes:
'''bash
git add -A && git commit -m "..." && git push
'''
