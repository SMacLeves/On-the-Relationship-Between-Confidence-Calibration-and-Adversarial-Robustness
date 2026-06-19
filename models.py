"""ResNet-18 architecture adapted for CIFAR (32x32) inputs."""
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes=100):
    """Return a CIFAR-adapted ResNet-18 with a 3x3 stride-1 stem and no max-pool.

    The standard torchvision stem (7x7 conv, stride 2 + max-pool) reduces
    32x32 inputs to 8x8 before the first residual stage, discarding spatial
    information too early and reduce dimension in the last step aggressively
    Therefore the modifications: they preserve the full 32x32 resolution
    through the stem, following standard practice in the CIFAR robustness literature
    """
    model = models.resnet18(weights=None)
    # CIFAR-friendly stem: 3x3 conv, no aggressive downsampling
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
