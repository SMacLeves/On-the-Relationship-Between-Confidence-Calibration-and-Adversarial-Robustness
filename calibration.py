"""Calibration metrics, reliability diagrams, and temperature scaling"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """Returns confidences, predictions, labels as numpy arrays"""
    model.eval()
    all_confs, all_preds, all_labels = [], [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_confs.append(confs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_confs).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


def collect_adversarial_predictions(model, dataloader, device, eps=8/255, alpha=2/255, steps=10):
    """
    Returns confidences, predictions, labels for PGD-perturbed inputs
    Use this to measure how well calibrated a model is under attack
    """
    from attacks import pgd_attack
    model.eval()
    all_confs, all_preds, all_labels = [], [], []

    for images, labels in dataloader:
        images  = images.to(device)
        labels_ = labels.to(device)

        with torch.enable_grad():
            adv = pgd_attack(model, images, labels_, device, eps=eps, alpha=alpha, steps=steps)

        with torch.no_grad():
            probs = torch.softmax(model(adv), dim=1)
            confs, preds = probs.max(dim=1)

        all_confs.append(confs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_confs).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


def compute_ece(confidences, predictions, labels, n_bins=15):
    """Expected Calibration Error (where the lower the better)"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / n


def reliability_diagram(confidences, predictions, labels, n_bins=15,
                        title="Reliability Diagram", save_path=None):
    """Plot a reliability diagram and return ECE 
    Saves to save_path if provided"""
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_centers = [(lo + hi) / 2 for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
    bin_accs    = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
        else:
            bin_accs.append((predictions[mask] == labels[mask]).mean())

    ece = compute_ece(confidences, predictions, labels, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_centers, bin_accs, width=1 / n_bins, alpha=0.7, label="Accuracy", align="center")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return ece


class TemperatureScaler(nn.Module):
    """Wraps a trained model and exposes a single learnable temperature T"""

    def __init__(self, model):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.model(x) / self.temperature

    @torch.no_grad()
    def _collect_logits(self, dataloader, device):
        self.model.eval()
        logits_list, labels_list = [], []
        for images, labels in dataloader:
            images = images.to(device)
            logits_list.append(self.model(images).cpu())
            labels_list.append(labels)
        return torch.cat(logits_list), torch.cat(labels_list)

    def fit(self, val_dl, device, lr=0.01, max_iter=50):
        """Minimise NLL on the validation set to find optimal T"""
        logits, labels = self._collect_logits(val_dl, device)
        logits = logits.to(device)
        labels = labels.to(device)
        criterion  = nn.CrossEntropyLoss()
        optimizer  = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self
