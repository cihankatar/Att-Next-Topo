# Att-Next-Topo

PyTorch project for semantic segmentation of skin lesion images using a custom ATTNext encoder-decoder architecture with optional topological loss and SSL modes. The project trains models, logs experiments with Weights & Biases (wandb), and produces segmentation maps.

This repository contains training, testing, data loading, augmentation, visualization, and model code used for skin lesion segmentation research/experiments.

## Quick overview
- Model: `models/Model.py` (ATTNext) — encoder (`models/enc.py`), decoder (`models/dec.py`) and a lightweight segmentation head.
- Data: `data/data_loader.py` and `data/Custom_Dataset.py` provide dataset+transforms and `loader(...)` factory.
- Training loop: `train.py` — uses Adam, CosineAnnealingLR, Dice+BCE loss and optional topological loss (`utils/Loss.py`).
- Testing / inference: `test.py` — loads checkpoints and logs predictions to wandb.
- Utilities: `utils/metrics.py`, `visualization.py`, `augmentation/Augmentation.py`.
- Experiment config & wandb helpers: `wandb_init.py` (parser + wandb init helper).

## Requirements
Recommended Python environment: 3.8+ (project was developed/tested with Python 3.9).

Create a virtual environment and install packages (GPU torch install is environment-specific):

```bash
# create env
python -m venv .venv
source .venv/bin/activate

# Install from requirements (see notes about torch below)
pip install -r requirements.txt
```

Notes:
- Install `torch`/`torchvision` matching your CUDA version following https://pytorch.org/get-started/locally. The `requirements.txt` lists a placeholder `torch` entry; if you need GPU acceleration use the command from PyTorch site.
- `gudhi` and `torch_topological` are required for the topological loss; if you don't use topological features you can skip them.

## Environment variables (required)
Set these before running train/test locally. Example for zsh:

```bash
export ML_DATA_ROOT=/path/to/datasets/   # root with dataset folders like isic_2018_1/
export ML_DATA_OUTPUT=/path/to/output/   # used when CUDA/GPU is available
export ML_DATA_OUTPUT_LOCAL=/tmp/att_next_output/  # used for local CPU runs
export WANDB_API_KEY="<your_wandb_api_key>"
export WANDB_DIR=/path/for/wandb
```

## Dataset layout expected by `data_loader.loader`
The loader expects dataset directories with the following structure under `$ML_DATA_ROOT`:

- isic_2018_1/
  - train/
    - images/   (*.jpg)
    - masks/    (*.png)
  - val/
    - images/
    - masks/
  - test/
    - images/
    - masks/

The code contains explicit mappings and file extension handling for: `isic_2018_1`, `kvasir_1`, `ham_1`, `PH2Dataset`, `isic_2016_1` in `data/data_loader.py`.

## How to run

Train (default supervised settings):

```bash
# make sure env vars are set
python train.py
```

Test / inference (loads checkpoint path inferred from model class name + args):

```bash
python test.py
```

Important run-time options can be changed via `wandb_init.parser_init(...)` or by passing CLI args (the parser is in `wandb_init.py`). Typical flags include `--op`, `--mode`, `--bsize`, `--epochs`, `--imsize`, etc. Example:

```bash
python train.py --bsize 8 --epochs 200 --mode supervised
```

## Where experiments are logged
- wandb is initialized in `wandb_init.py`. Make sure `WANDB_API_KEY` is set. For local CPU runs the repo uses a different project name (`Temp_Att-Next-SSL_local`).

## Checkpoints
- Checkpoint filename is constructed in `train.py` / `test.py` as `folder_path + model.__class__.__name__ + res` where `res` is a string with the parser argument key-value pairs. `folder_path` is derived from `ML_DATA_OUTPUT`/`ML_DATA_OUTPUT_LOCAL` and the dataset folder.

## Project-specific conventions & patterns (so an AI helper or new dev can be productive)
- Configuration is centralized in `wandb_init.py` via `config_func()` and `parser_init()` — prefer to change defaults there for reproducible experiments.
- Data loader expects normalized 0-1 inputs and returns masks as single-channel tensors (masks expanded to 1 channel in `Custom_Dataset.dataset`). The dataset uses synchronized RNG seeds to apply identical transforms to images and masks.
- Mode behavior: `mode` may be `ssl`, `supervised`, or `ssl_pretrained`. When `ssl_pretrained` is used, the encoder is expected to be frozen and `wandb_init` may include SSL config.
- Augmentations: `augmentation/Augmentation.py` includes Cutout, Cutmix and other augmentations; `data/data_loader.py` uses torchvision v2 transforms (v2 API). Pay attention to `v2.Compose` usage.

## Files to look at when debugging or extending
- Model: `models/Model.py`, `models/enc.py`, `models/dec.py`
- Training loop: `train.py`
- Testing/inference: `test.py`
- Data loading: `data/data_loader.py`, `data/Custom_Dataset.py`
- Losses (Dice + BCE + Topological): `utils/Loss.py`
- Metrics: `utils/metrics.py`
- Augmentations: `augmentation/Augmentation.py`
- Experiment config/wandb: `wandb_init.py`

## Troubleshooting
- If dataset files are not found: verify `ML_DATA_ROOT` and dataset folder names. `data_loader.py` uses `glob()` with the subfolders `train/val/test` and will sort file lists.
- If wandb fails to init: check `WANDB_API_KEY`, run `wandb login <key>` and ensure `WANDB_DIR` is valid.
- GPU/torch issues: install `torch` according to your CUDA version from pytorch.org; otherwise run on CPU.
- If topological loss imports fail: ensure `gudhi` and `torch_topological` are installed; these can be tricky to install on some platforms — consider disabling topological loss (set `addtopoloss=False` in `train.py`) while debugging.

## Suggested next steps / improvements
- Add a small dataset example (few images) under a `data/sample/` folder and a quick-start script to train for 1 epoch for CI/local tests.
- Add `pytest` unit tests for dataset loader and metric functions.

---
If anything in this README is unclear or you'd like me to include a short example `train` command with specific args (for example exact SSL or supervised settings you used), tell me which mode & dataset you used and I'll add it.
