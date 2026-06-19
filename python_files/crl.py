"""Correctness Ranking Loss training loops (Moon et al., 2020)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def negative_entropy(logits, normalize=False, max_value=None):
    """Negative entropy as confidence proxy, the higher, the more confident)"""
    softmax     = F.softmax(logits, dim=1)
    log_softmax = F.log_softmax(logits, dim=1)
    entropy = -(softmax * log_softmax).sum(dim=1)
    if normalize:
        return -entropy / max_value
    return -entropy


class History:
    """
    Tracks per-input historical correctness across training epochs
    Adapted from Moon et al. (2020)
    """

    def __init__(self, n_data):
        self.correctness     = np.zeros(n_data)
        self.confidence      = np.zeros(n_data)
        self.max_correctness = 1

    def correctness_update(self, data_idx, correctness, output):
        probs = F.softmax(output, dim=1)
        conf, _ = probs.max(dim=1)
        idx = data_idx.cpu().numpy()
        self.correctness[idx] += correctness.cpu().numpy()
        self.confidence[idx]   = conf.cpu().detach().numpy()

    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    def _normalize(self, data):
        lo = self.correctness.min()
        hi = float(self.max_correctness)
        return (data - lo) / (hi - lo + 1e-8)

    def get_target_margin(self, idx1, idx2, device):
        c1 = self._normalize(self.correctness[idx1.cpu().numpy()])
        c2 = self._normalize(self.correctness[idx2.cpu().numpy()])
        greater = (c1 > c2).astype(np.float32)
        less    = (c1 < c2).astype(np.float32) * -1
        target  = torch.tensor(greater + less, dtype=torch.float32, device=device)
        margin  = torch.tensor(np.abs(c1 - c2), dtype=torch.float32, device=device)
        return target, margin


def crl_train_one_epoch(model, dataloader, criterion, optimizer, history, device,
                        epoch=0, rank_target='softmax', rank_weight=1.0, num_classes=100):
    """
    One epoch of CRL training (Moon et al., 2020)
    dataloader must yield (images, labels, indices), so wrap train_subset with IndexedDataset
    """
    model.train()
    ranking_criterion = nn.MarginRankingLoss(margin=0.0)
    running_loss = cls_running = rank_running = 0.0
    correct = total = 0

    for images, labels, idx in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        idx    = idx.to(device)

        output = model(images)

        # confidence measure
        if rank_target == 'softmax':
            probs = F.softmax(output, dim=1)
            confidence, _ = probs.max(dim=1)
        elif rank_target == 'entropy':
            confidence = negative_entropy(output, normalize=True, max_value=np.log(num_classes))
        elif rank_target == 'margin':
            top2, _ = torch.topk(F.softmax(output, dim=1), 2, dim=1)
            confidence = top2[:, 0] - top2[:, 1]
        else:
            raise ValueError(f"rank_target must be 'softmax', 'entropy', or 'margin'")

        # ranking pairs (circular shift)
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2        = torch.roll(idx, -1)

        rank_target_val, rank_margin = history.get_target_margin(idx, idx2, device)
        rank_target_nz = rank_target_val.clone()
        rank_target_nz[rank_target_nz == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nz

        ranking_loss = ranking_criterion(rank_input1, rank_input2, rank_target_val)

        cls_loss = criterion(output, labels)
        loss     = cls_loss + rank_weight * ranking_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds    = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        running_loss += loss.item()          * images.size(0)
        cls_running  += cls_loss.item()      * images.size(0)
        rank_running += ranking_loss.item()  * images.size(0)

        history.correctness_update(idx, (preds == labels).float(), output)

    history.max_correctness_update(epoch)
    return running_loss / total, correct / total


def madry_crl_train_one_epoch(model, dataloader, criterion, optimizer, history, device,
                               epoch=0, rank_target='softmax', rank_weight=1.0,
                               num_classes=100, eps=8/255, alpha=2/255, steps=5):
    """
    Madry adversarial training + CRL ranking loss
    CE loss is on PGD adversarial examples, CRL ranking uses clean predictions
    dataloader must yield (images, labels, indices), so wrap train_subset with IndexedDataset
    """
    from torch.cuda.amp import GradScaler, autocast
    from attacks import pgd_attack

    scaler = GradScaler(enabled=device.type == 'cuda')
    model.train()
    ranking_criterion = nn.MarginRankingLoss(margin=0.0)
    running_loss = 0.0
    correct = total = 0

    for images, labels, idx in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        idx    = idx.to(device)

        # PGD in full precision — AMP inside attack causes instability
        adv = pgd_attack(model, images, labels, device, eps=eps, alpha=alpha, steps=steps)

        model.train()
        optimizer.zero_grad()
        with autocast(enabled=device.type == 'cuda'):
            adv_output   = model(adv)
            cls_loss     = criterion(adv_output, labels)
            clean_output = model(images)

        # Ranking in float32 to avoid dtype issues with history tensors
        clean_fp32 = clean_output.float()
        if rank_target == 'softmax':
            probs = F.softmax(clean_fp32, dim=1)
            confidence, _ = probs.max(dim=1)
        elif rank_target == 'entropy':
            confidence = negative_entropy(clean_fp32, normalize=True, max_value=np.log(num_classes))
        elif rank_target == 'margin':
            top2, _ = torch.topk(F.softmax(clean_fp32, dim=1), 2, dim=1)
            confidence = top2[:, 0] - top2[:, 1]
        else:
            raise ValueError(f"rank_target must be 'softmax', 'entropy', or 'margin'")

        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2        = torch.roll(idx, -1)

        rank_target_val, rank_margin = history.get_target_margin(idx, idx2, device)
        rank_target_nz = rank_target_val.clone()
        rank_target_nz[rank_target_nz == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nz

        ranking_loss = ranking_criterion(rank_input1, rank_input2, rank_target_val)
        loss = cls_loss.float() + rank_weight * ranking_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds    = adv_output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        running_loss += loss.item() * images.size(0)

        history.correctness_update(idx, (clean_output.argmax(dim=1) == labels).float(), clean_output)

    history.max_correctness_update(epoch)
    return running_loss / total, correct / total
