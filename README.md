# COMP 457 — Advanced AI

Coursework for Athabasca COMP 457. Code is shared via git; each machine
builds its own local conda environment (envs contain platform-specific
binaries and are never committed).

## Setup

Prereq: Miniconda or Anaconda installed
(https://docs.conda.io/en/latest/miniconda.html).

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

### 3. Install PyTorch (platform-specific)

PyTorch ships different binaries per platform. Confirm the current
command at https://pytorch.org/get-started/locally/ before running.

**macOS (Apple Silicon uses MPS, Intel is CPU-only)**

```bash
pip install torch torchvision
```

**Windows with NVIDIA GPU** — check your CUDA version with `nvidia-smi`,
then pick the matching index URL. Example for CUDA 12.4:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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

## Day-to-day workflow between machines

Before starting work on either machine:
'''bash
git pull
'''

After making changes:
'''bash
git add -A && git commit -m "..." && git push
'''
