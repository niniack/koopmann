import os
from pathlib import Path
from typing import Optional, Union

import fire
import numpy as np
import torch
import torchattacks
import wandb
from config_def import Config
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from koopmann.log import logger
from koopmann.models import MLP, ResMLP, resnet18, resnet18_mnist
from koopmann.utils import get_device
from scripts.utils import (
    get_dataloaders,
    get_lr_schedule,
    get_optimizer,
    setup_config,
)


def get_model(config, dataset):
    if config.model.conv:
        if "CIFAR10" in dataset.__class__.__name__:
            model = resnet18()
        elif "MNIST" in dataset.__class__.__name__:
            model = resnet18_mnist()
        else:
            raise NotImplementedError()

    else:
        if config.model.residual:
            model = ResMLP(
                in_features=np.prod(dataset.in_features),
                out_features=dataset.out_features,
                hidden_config=config.model.hidden_neurons,
                bias=config.model.bias,
                batchnorm=config.model.batchnorm,
                nonlinearity="relu",
            )
        else:
            model = MLP(
                in_features=np.prod(dataset.in_features),
                out_features=dataset.out_features,
                hidden_config=config.model.hidden_neurons,
                bias=config.model.bias,
                batchnorm=config.model.batchnorm,
                nonlinearity="relu",
            )

    return model


def save_model(model, config, dataset_name):
    if not config.save_dir:
        return None

    os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
    model_path = Path(config.save_dir)

    suffix = config.suffix if config.suffix else ""
    model.save_model(model_path, suffix=suffix, dataset=dataset_name)


def compute_model_stats(model, step, log_histograms=False):
    """
    Log model statistics to wandb.
    """
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
                    stats[f"weights/{name}/histogram"] = wandb.Histogram(param.data.cpu().numpy())

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

    return {
        "clean_accuracy": 100 * clean_correct / total,
    }


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
    epoch: int,
) -> dict:
    model.to(device).train()

    metrics = {"train/loss": 0.0, "train/accuracy": 0.0}

    num_batches = len(train_loader)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move data to device
        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).squeeze(),
        )
        batch_size = inputs.size(0)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        metrics["train/loss"] += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        metrics["train/accuracy"] += correct / batch_size

    # # Calculate curvature at specified frequency
    # if measure_curvature and (epoch + 1) % curvature_frequency == 0:
    #     metrics["curvature"] = calculate_curvature(model, train_loader, device, criterion)

    # Normalize metrics by number of batches
    metrics["train/loss"] /= num_batches
    metrics["train/accuracy"] /= num_batches
    metrics["epoch"] = epoch

    return metrics


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
    # Config
    config = setup_config(config_path_or_obj, Config)
    device = get_device()

    # Data
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(config)

    # Model
    model = get_model(config, train_dataset)
    model.to(device).train().hook_model()

    # Loss
    loss = nn.CrossEntropyLoss()

    # Optim
    optimizer = get_optimizer(config, model)

    # Scheduler
    # scheduler = None
    scheduler = get_lr_schedule(
        lr_schedule_type="cyclic",
        n_epochs=config.optim.num_epochs,
        lr_max=config.optim.learning_rate,
        optimizer=optimizer,
    )

    metrics = {}
    for epoch in range(config.optim.num_epochs):
        metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
            epoch=epoch,
        )

        # Log metrics
        if scheduler:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = config.optim.learning_rate
        metrics["lr"] = lr

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            # Model statistics
            model_stats = compute_model_stats(model, epoch, log_histograms=True)
            metrics.update(model_stats)

            # Neural collapse stats
            # nc_stats = compute_neural_collapse_metrics(model, config, test_loader, device)
            # metrics.update(nc_stats)

            # curvature = compute_curvature(
            #     model=model,
            #     dataloader=test_loader,
            #     device=device,
            # )
            # metrics.update({"curvature": curvature})

            eval_metrics = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
            )

            metrics.update(eval_metrics)

            # Print out
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, Loss: {metrics['train/loss']:.4f}, "
            )

        # Log all epoch metrics
        if (epoch + 1) % config.print_freq // 2 == 0:
            wandb.log(metrics, step=epoch)

    # Save model
    if config.save_dir:
        save_model(model=model, config=config, dataset_name=test_dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
