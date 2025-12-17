import torch
from torch.utils.data import DataLoader

import wandb
from koopmann.data import get_dataset_class


def get_dataloaders(
    config,
    train_batch_size=None,
    test_batch_size=None,
    shuffle=True,
    train_subset=None,
    test_subset=None,
):
    # Train data
    train_size = train_batch_size if train_batch_size else config.optim.batch_size
    DatasetClass = get_dataset_class(name=config.train_data.dataset_name)
    train_dataset = DatasetClass(config=config.train_data, root=config.train_data.root)  # type: ignore

    if train_subset:
        subset_indices = list(range(0, train_subset))
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_size,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        prefetch_factor=4,
    )

    # Test data
    test_size = test_batch_size if test_batch_size else config.optim.batch_size
    original_split = config.train_data.split
    config.train_data.split = "test"
    test_dataset = DatasetClass(config=config.train_data, root=config.train_data.root)  # type: ignore
    config.train_data.split = original_split

    if test_subset:
        subset_indices = list(range(0, test_subset))
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_size,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        prefetch_factor=4,
    )

    return train_loader, test_loader, train_dataset, test_dataset


def compute_model_stats(model, step, log_histograms=False):
    model.eval()

    stats = {}

    # Iterate through named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Only include weights to reduce clutter
            if "weight" in name:
                # Log gradient norms if they exist
                if param.grad is not None:
                    stats[f"gradients/{name}/norm"] = param.grad.norm().item()

                # Log weight norms
                stats[f"weights/{name}/norm"] = param.norm().item()

                # Log histograms (more expensive operation)
                if log_histograms:
                    stats[f"weights/{name}/histogram"] = wandb.Histogram(
                        param.data.cpu().numpy()
                    )

    return stats


def evaluate_model(model, dataloader, device):
    model.eval()

    clean_correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        batch_size = inputs.size(0)
        total += batch_size

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()

    return {"test/accuracy": 100 * clean_correct / total}
