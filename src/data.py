from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_cifar10_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
    train_subset: int | None = 1000,
    num_workers: int = 2,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    if train_subset is not None and train_subset > 0:
        train_set = Subset(train_set, range(min(train_subset, len(train_set))))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
