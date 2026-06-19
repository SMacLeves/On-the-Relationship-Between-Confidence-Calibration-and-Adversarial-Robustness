"""CIFAR-10/100 data loading, normalisation statistics, and dataset utilities"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as tt

CIFAR10_STATS  = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
CIFAR100_STATS = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

_MANUAL_DOWNLOAD = {
    "cifar10":  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps any Dataset and appends the sample index as a third return value
    Required by CRL training, which needs (image, label, index) per batch
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx

    def __len__(self):
        return len(self.dataset)


def _direct_download(name, root):
    """Bypass torchvision and download + extract the tarball directly via urllib"""
    import urllib.request
    import tarfile
    import time

    url  = _MANUAL_DOWNLOAD[name.lower()]
    fname = url.split("/")[-1]
    dest  = os.path.join(root, fname)
    os.makedirs(root, exist_ok=True)

    if not os.path.exists(dest):
        for attempt in range(3):
            try:
                print(f"  Attempt {attempt + 1}/3: downloading {fname} …")
                urllib.request.urlretrieve(url, dest)
                print("  Download complete.")
                break
            except Exception as err:
                print(f"  Failed: {err}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"\nAll 3 download attempts failed for {name.upper()}.\n"
                        f"Please download manually:\n"
                        f"  {url}\n"
                        f"and place the file in: {os.path.abspath(root)}\n"
                        f"then re-run this cell."
                    ) from err
    else:
        print(f"  Tarball already on disk: {dest}")

    print("  Extracting …")
    with tarfile.open(dest, "r:gz") as t:
        t.extractall(root)
    print("  Extraction complete.")


def _safe_load(DatasetClass, name, **kwargs):
    """Load a torchvision dataset, and fall back to direct urllib download on HTTP 503"""
    try:
        return DatasetClass(**kwargs)
    except Exception as e:
        is_503 = any(tok in str(e) for tok in ("503", "Service Unavailable", "URLError", "HTTP Error"))
        if not is_503:
            raise

        root = kwargs.get("root", "./data")
        print(f"[HTTP 503] torchvision download failed for {name.upper()}. Trying direct download …")
        _direct_download(name, root)

        # Retry without download now that the files are on disk
        kw = {**kwargs, "download": False}
        return DatasetClass(**kw)


def get_cifar_loaders(dataset_name="cifar100", batch_size=128, val_fraction=0.1, seed=42):
    """Load CIFAR-10 or CIFAR-100 and return train/val/test datasets and dataloaders

    Carves val_fraction of the training set as a validation split (no augmentation)
    Training set receives RandomCrop(32, padding=4) and RandomHorizontalFlip(p=0.5)
    Returns (train_subset, val_dataset, test_dataset, train_dl, val_dl, test_dl, class_names, stats, num_classes)
    """
    if dataset_name.lower() == "cifar10":
        DatasetClass = CIFAR10
        num_classes  = 10
        stats        = CIFAR10_STATS
    elif dataset_name.lower() == "cifar100":
        DatasetClass = CIFAR100
        num_classes  = 100
        stats        = CIFAR100_STATS
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    train_tfms = tt.Compose([
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
        tt.Normalize(*stats),
    ])
    test_tfms = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(*stats),
    ])

    full_train   = _safe_load(DatasetClass, dataset_name, root='./data', train=True,  download=True, transform=train_tfms)
    test_dataset = _safe_load(DatasetClass, dataset_name, root='./data', train=False, download=True, transform=test_tfms)

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_subset, val_indices_wrapper = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Validation set uses test transforms (no augmentation)
    val_base   = DatasetClass(root='./data', train=True, download=False, transform=test_tfms)
    val_dataset = torch.utils.data.Subset(val_base, val_indices_wrapper.indices)

    train_dl = DataLoader(train_subset,  batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_subset, val_dataset, test_dataset, train_dl, val_dl, test_dl, full_train.classes, stats, num_classes
