# COMP 457 — Advanced AI

Coursework for Athabasca COMP 457. Code is shared via git; each machine
builds its own local Python environment (envs contain platform-specific
binaries and are never committed).

## Setup

### 1. Clone

```bash
git clone <repo-url> Comp457_Advanced_AI
cd Comp457_Advanced_AI
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install PyTorch

PyTorch installs differ by platform because of GPU support. Use the
picker at https://pytorch.org/get-started/locally/ to confirm the
current command, then:

**macOS (Apple Silicon uses MPS, Intel is CPU-only)**

```bash
pip install torch torchvision
```

**Windows with NVIDIA GPU (CUDA 12.4 shown — check `nvidia-smi` for your version)**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install the rest

```bash
pip install -r requirements.txt
```

### 5. Verify

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available(), 'mps:', torch.backends.mps.is_available())"
```

## Run the Unit 1 quickstart

```bash
python src/unit1_quickstart.py
```

Downloads FashionMNIST to `data/`, trains a small MLP for 5 epochs,
saves weights to `model.pth`, then reloads them for one prediction.
