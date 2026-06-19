# On the Relationship Between Confidence Calibration and Adversarial Robustness in ResNet Models

BSc Artificial Intelligence thesis — Radboud University, 2026  
**Author:** Levente Olivér Bódi  
**Supervisor:** Dr. Louis ten Bosch  
**Second reader:** Dr. Luca Ambrogioni

---

## Overview

This repository contains the code for the experiments in the thesis. Six training configurations are evaluated on a CIFAR-adapted ResNet-18 across four metrics: clean accuracy, PGD-10 adversarial accuracy, clean ECE, and adversarial ECE.

| Method | Description |
|---|---|
| ERM Baseline | Standard cross-entropy on clean inputs |
| ERM + Temperature Scaling | Post-hoc logit rescaling (Guo et al., 2017) |
| ERM + CRL | Correctness Ranking Loss (Moon et al., 2020) |
| Madry | PGD-7 adversarial training (Madry et al., 2018) |
| CCAT | Adversarial training + KL penalty toward uniform (Stutz et al., 2020) |
| Madry + CRL | Novel combination of Madry adversarial training and CRL |

A post-hoc Mahalanobis-distance-based risk layer is additionally evaluated on three of the trained models.

---

## Repository Structure

```
├── Thesis.ipynb              # Main experiment notebook (run cells top to bottom)
├── requirements.txt          # Pythong package requirements
├── python_files/             # Folder to store python files used in the Thesis.ipynb
    ├── models.py             # CIFAR-adapted ResNet-18
    ├── train.py              # Clean training and evaluation loops
    ├── data.py               # Data loading and normalisation
    ├── calibration.py        # ECE, reliability diagrams, temperature scaling
    ├── attacks.py            # PGD attack and adversarial training loops
    ├── crl.py                # Correctness Ranking Loss training loops
    └── outputs/              # Generated models, plots, and CSVs (not versioned)
        └── cifar10/
            └── 2026-05-04_13-55-45/   # Final experimental run
                ├── models/
                ├── plots/
                └── csv/
```

The `confidence-aware-learning-master/` folder contains the original CRL code by Moon et al(2020), included for reference. It is not used directly by the notebook.

---

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for adversarial training. The notebook falls back to CPU automatically but training will be significantly slower.

---

## Running the Experiments

Open `Thesis.ipynb` and run cells sequentially. Each training block is labelled with the experiment name (e.g. `ERM Baseline`). Outputs (model checkpoints, plots, CSVs) are saved automatically to a timestamped folder under `outputs/cifar10/`.

Key hyperparameters (all in the notebook):

| Parameter | Value |
|---|---|
| Architecture | ResNet-18 (CIFAR-adapted) |
| Dataset | CIFAR-10 |
| Epochs | 30 |
| Optimiser | Adam, lr=1e-3, weight decay=1e-4 |
| LR schedule | CosineAnnealingLR |
| PGD ε | 8/255 (L-inf) |
| PGD α | 2/255 |
| PGD steps (train) | 7 |
| PGD steps (eval) | 10 |

---

## Results Summary

| Method | Clean acc. | PGD-10 acc. | ECE clean | ECE adv |
|---|---|---|---|---|
| ERM Baseline | 92.79% | 3.81% | 0.044 | 0.952 |
| ERM + Temp. Scaling | 92.79% | 3.81% | 0.037 | 0.952 |
| ERM + CRL | 92.29% | 6.70% | 0.019 | 0.893 |
| Madry | 90.65% | 75.98% | 0.028 | 0.140 |
| **Madry + CRL** | **90.59%** | **77.17%** | **0.018** | **0.072** |
| CCAT (λ=1.0) | 15.94% | 3.41% | 0.095 | 0.215 |

CCAT results are not comparable due to training collapse at λ=1.0.
