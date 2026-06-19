"""PGD attack and adversarial training loops (Madry et al., 2018; Stutz et al., 2020)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(model, images, labels, device, eps=8/255, alpha=2/255, steps=10, random_start=True):
    """
    PGD L-inf attack (Madry et al., 2018)
    Images are assumed to be already normalised; perturbation is bounded to [-eps, eps]
    """
    model.eval()
    images    = images.to(device)
    labels    = labels.to(device)
    criterion = nn.CrossEntropyLoss()

    delta = torch.empty_like(images).uniform_(-eps, eps) if random_start else torch.zeros_like(images)
    delta = delta.to(device)

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = criterion(model(images + delta), labels)
        loss.backward()
        with torch.no_grad():
            delta = (delta + alpha * delta.grad.sign()).clamp(-eps, eps)

    return (images + delta).detach()


@torch.no_grad()
def evaluate_clean_and_pgd(model, dataloader, device, eps=8/255, alpha=2/255, steps=10):
    """Returns (clean_acc, pgd_acc)"""
    model.eval()
    clean_correct = pgd_correct = total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        clean_preds = model(images).argmax(dim=1)
        clean_correct += (clean_preds == labels).sum().item()

        # PGD accuracy, temporarily enable gradients
        with torch.enable_grad():
            adv = pgd_attack(model, images, labels, device, eps=eps, alpha=alpha, steps=steps)
        pgd_preds = model(adv).argmax(dim=1)
        pgd_correct += (pgd_preds == labels).sum().item()

        total += labels.size(0)

    return clean_correct / total, pgd_correct / total


def adversarial_train_one_epoch(model, dataloader, criterion, optimizer, device,
                                eps=8/255, alpha=2/255, steps=7):
    """One epoch of Madry adversarial training with AMP for speed"""
    from torch.cuda.amp import GradScaler, autocast
    scaler  = GradScaler(enabled=device.type == 'cuda')
    model.train()
    running_loss = 0.0
    correct = total = 0

    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        adv = pgd_attack(model, images, labels, device, eps=eps, alpha=alpha, steps=steps)

        model.train()
        optimizer.zero_grad()
        with autocast(enabled=device.type == 'cuda'):
            outputs = model(adv)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


def ccat_train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes,
                         eps=8/255, alpha=2/255, steps=5, lambda_adv=1.0):
    """
    Trains CCAT, CE on clean examples + KL(adversarial output, uniform) (Stutz et al., 2020).
    Default steps=5, which is sufficient for training, so use 10 only for evaluation
    AMP enabled on CUDA for 1.5x additional speedup
    """
    from torch.cuda.amp import GradScaler, autocast
    scaler         = GradScaler(enabled=device.type == 'cuda')
    model.train()
    running_loss   = 0.0
    correct = total = 0
    uniform_target = torch.full((1, num_classes), 1.0 / num_classes)

    for images, labels in dataloader:
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        uniform = uniform_target.expand(images.size(0), -1).to(device)

        # PGD runs in full precision — AMP inside the attack causes instability
        adv = pgd_attack(model, images, labels, device, eps=eps, alpha=alpha, steps=steps)

        model.train()
        optimizer.zero_grad()
        with autocast(enabled=device.type == 'cuda'):
            out_clean  = model(images)
            loss_clean = criterion(out_clean, labels)
            log_probs  = F.log_softmax(model(adv), dim=1)
            loss_adv   = F.kl_div(log_probs, uniform, reduction='batchmean')
            loss       = loss_clean + lambda_adv * loss_adv

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds         = out_clean.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total
