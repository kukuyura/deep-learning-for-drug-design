# EGNN for Molecular Property Prediction (QM9 / HOMO)

Hands-on reproduction of an **E(n) Equivariant Graph Neural Network** for predicting molecular orbital energy (HOMO) on the [QM9](http://quantum-machine.org/datasets/) dataset.

Based on the paper:

> Satorras, Hoogeboom, Welling — *E(n) Equivariant Graph Neural Networks* ([arXiv:2102.09844](https://arxiv.org/pdf/2102.09844.pdf))

and the educational implementation [senya-ashukha/simple-equivariant-gnn](https://github.com/senya-ashukha/simple-equivariant-gnn) (data loaders + reference training setup).

## Why this is interesting

EGNNs respect Euclidean symmetries (rotations / translations) that matter for 3D molecular graphs. That makes them a natural architecture family for **drug / molecule property prediction**, where atom coordinates are part of the input.

## What the notebook does

1. Loads QM9 via the reference data utilities (target: **HOMO** energy).
2. Implements `ConvEGNN` / `NetEGNN` in PyTorch (message passing with pairwise distances, edge inference, graph pooling).
3. Trains with Adam + cosine LR schedule, MAD-normalized L1 loss.
4. Tracks train / val / test MAE (meV-scale reporting as in the reference setup).

## Reported numbers (from notebook comments)

| Checkpoint | Val | Test |
|------------|-----|------|
| Early overfit point (~epoch 280) | 42.2 | 42.7 |
| Better run cited in notebook | 30.2 | 30.9 |

Lower is better. Exact reproducibility depends on hardware / seeds / full training length (up to 1000 epochs in the script).

## Repo layout

```
egnn.ipynb   # model definition + training loop
```

The notebook clones `simple-equivariant-gnn` for QM9 loaders — keep network access when running.

## How to run

```bash
# CUDA-capable GPU recommended
pip install torch numpy
jupyter notebook egnn.ipynb
```

## Honesty / scope

This is a **learning / reproduction** project, not a novel architecture paper. Value for a portfolio: shows comfort with geometric deep learning, PyTorch GNN code, and a chemistry-relevant benchmark — not a claim of SOTA drug-design research.
