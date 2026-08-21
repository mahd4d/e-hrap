# Enhanced Hybrid Redundancy-Aware Pruning (E-HRAP)

This repository contains a reproducible PyTorch implementation of Enhanced
Hybrid Redundancy-Aware Pruning (E-HRAP), a structured filter-pruning method
for convolutional neural networks.

E-HRAP separates pruning into two stages:

1. **Adaptive layer-budget allocation.** Gradient sensitivity protects
   important layers and assigns a larger pruning budget to less-sensitive
   layers.
2. **Hybrid filter ranking.** Normalized magnitude significance is combined
   with activation uniqueness, derived from inter-filter correlation, to rank
   filters within each layer.

The implementation performs physical channel removal and then fine-tunes the
compressed network. It also reports accuracy, parameter reduction, MAC
reduction, latency, speedup, and recovery-efficiency measurements.

## Experimental coverage

The full grid evaluates:

- **Datasets:** CIFAR-10, CIFAR-100, SVHN, FashionMNIST, and MNIST
- **Architectures:** SimpleCNN, VGG-11-BN, and CIFAR ResNet-20
- **Methods:** Uniform-Magnitude, Adaptive-Magnitude,
  Adaptive-Correlation, and E-HRAP
- **Target pruning rates:** 10%, 20%, 30%, 40%, and 50%
- **Reference seed:** 42

This produces 300 dataset-architecture-method-rate configurations.

## Repository contents

| Path | Description |
| --- | --- |
| `s9.py` | Complete training, pruning, fine-tuning, evaluation, and export pipeline |
| `requirements.txt` | Python package requirements |
| `results/experiment_manifest.json` | Hardware, software, and run configuration |
| `results/full_grid_results.csv` | Per-configuration measurements |
| `results/full_grid_summary.csv` | Aggregated experimental results |

Datasets and model checkpoints are not stored in the repository. Supported
datasets are downloaded automatically by torchvision unless `--no-download`
is supplied.

## Environment

The reported full-grid experiment was executed with:

- NVIDIA RTX 3080 Ti
- 32 GB RAM
- 16 vCPUs
- PyTorch 2.6.0 with CUDA 12.6
- torchvision 0.21.0 with CUDA 12.6

Python 3.10 or newer is recommended.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install a CUDA-compatible PyTorch build following the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then
install the remaining requirements:

```bash
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Running the experiments

### Quick pipeline check

Use this first to verify dataset loading, pruning, fine-tuning, and result
export:

```bash
python s9.py --quick
```

### Full experimental grid

```bash
python s9.py --full-grid
```

The full run uses the five datasets, three architectures, four pruning
methods, five pruning rates, and seed 42.

### Focused run

```bash
python s9.py \
  --datasets CIFAR10 CIFAR100 \
  --architectures ResNet20 VGG11BN \
  --methods E-HRAP Adaptive-Magnitude \
  --rates 0.2 0.3 0.4 \
  --seeds 42
```

### JupyterLab

From a notebook cell:

```python
%run s9.py --quick
```

For the complete grid:

```python
%run s9.py --full-grid
```

Long runs should be started from a terminal or persistent session so they
continue if the browser disconnects.

## Main command-line options

```text
--full-grid
--datasets CIFAR10 CIFAR100 SVHN FashionMNIST MNIST
--architectures SimpleCNN VGG11BN ResNet20
--methods Uniform-Magnitude Adaptive-Magnitude Adaptive-Correlation E-HRAP
--rates 0.1 0.2 0.3 0.4 0.5
--seeds 42
--data-root ./data
--output-dir ./pruning_results/full_grid
--batch-size 128
--workers 2
--baseline-epochs 15
--fine-tune-epochs 10
--alpha 0.7
--beta 0.3
```

Run `python s9.py --help` for the complete argument list.

## Outputs

The selected output directory contains:

- `experiment_manifest.json`: run configuration and environment metadata
- `full_grid_results.csv`: one record for each experimental configuration
- `full_grid_summary.csv`: grouped mean and standard-deviation statistics
- `experiment.log`: console output captured during execution

Results are saved incrementally after every configuration, so a partial run
still retains completed measurements.

## Reproducibility notes

- Random seeds are applied to Python, NumPy, PyTorch, CUDA, and DataLoader
  workers.
- cuDNN deterministic execution is enabled and benchmarking is disabled.
- Baseline weights are shared across pruning methods within each matched
  dataset-architecture-seed setting.
- Latency is hardware- and software-dependent and should not be inferred from
  MAC reduction alone.
- The included reference results use one seed; multi-seed studies are
  recommended when estimating variance or statistical significance.
