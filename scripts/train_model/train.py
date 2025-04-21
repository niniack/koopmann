import os
from pathlib import Path
from typing import Optional, Union

import fire
import numpy as np
import torch
import torchattacks
from config_def import Config
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import wandb
from koopmann.log import logger
from koopmann.models import MLP, ResMLP, resnet18, resnet18_mnist
from koopmann.utils import get_device
from scripts.utils import (
    compute_curvature,
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
    suffix = suffix + "_adv" if config.adv.use_adversarial_training else suffix
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


def evaluate_adversarial_robustness(
    model, dataloader, device, dataset_mean, dataset_std, epsilon=(8 / 255)
):
    model.eval()

    # Initialize attacks
    fgsm_torch_attack = torchattacks.FGSM(model, eps=epsilon)
    fgsm_torch_attack.set_normalization_used(mean=dataset_mean, std=dataset_std)

    pgd_torch_attack = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 4, steps=40)
    pgd_torch_attack.set_normalization_used(mean=dataset_mean, std=dataset_std)

    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        batch_size = inputs.size(0)
        total += batch_size

        # Clean accuracy
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()

        # FGSM accuracy
        fgsm_inputs = fgsm_torch_attack(inputs, labels)
        with torch.no_grad():
            fgsm_outputs = model(fgsm_inputs)
            _, fgsm_predicted = fgsm_outputs.max(1)
            fgsm_correct += fgsm_predicted.eq(labels).sum().item()

        # PGD accuracy
        pgd_inputs = pgd_torch_attack(inputs, labels)
        with torch.no_grad():
            pgd_outputs = model(pgd_inputs)
            _, pgd_predicted = pgd_outputs.max(1)
            pgd_correct += pgd_predicted.eq(labels).sum().item()

    return {
        "clean_accuracy": 100 * clean_correct / total,
        "fgsm_accuracy": 100 * fgsm_correct / total,
        "pgd_accuracy": 100 * pgd_correct / total,
    }


def train_one_epoch_adversarial(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
    epoch: int,
    epsilon: float = (8 / 255),
    use_mixed_batch: bool = False,
    mixed_ratio: float = 0.5,
    dataset_mean=None,
    dataset_std=None,
) -> dict:
    model.to(device).train()
    metrics = {
        "train/loss": 0.0,
        "train/accuracy": 0.0,
        "train/adv_loss": 0.0,
        "train/adv_accuracy": 0.0,
    }
    num_batches = len(train_loader)

    # Create FGSM attack with normalization
    fgsm_attack = torchattacks.FGSM(model, eps=epsilon)
    fgsm_attack.set_normalization_used(mean=dataset_mean, std=dataset_std)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        batch_size = inputs.size(0)

        # Clean forward pass
        outputs = model(inputs)
        clean_loss = criterion(outputs, labels)

        # Get adversarial examples using torchattacks (handles normalization internally)
        adv_inputs = fgsm_attack(inputs, labels)

        # If using mixed batch, create the mixed input
        if use_mixed_batch:
            # Two separate forward passes instead of mixing the batch
            # Forward pass on clean examples
            optimizer.zero_grad()
            clean_outputs = model(inputs)
            clean_batch_loss = criterion(clean_outputs, labels)
            clean_batch_loss.backward()

            # Forward pass on adversarial examples
            adv_outputs = model(adv_inputs)
            adv_batch_loss = criterion(adv_outputs, labels)
            adv_batch_loss.backward()

            # Combined loss for metrics
            adv_loss = mixed_ratio * adv_batch_loss + (1 - mixed_ratio) * clean_batch_loss
        else:
            # Forward pass on only adversarial examples
            optimizer.zero_grad()
            adv_outputs = model(adv_inputs)
            adv_loss = criterion(adv_outputs, labels)
            adv_loss.backward()

        optimizer.step()

        # Calculate metrics
        _, predicted = outputs.max(1)
        clean_correct = predicted.eq(labels).sum().item()

        _, adv_predicted = adv_outputs.max(1)
        adv_correct = adv_predicted.eq(labels).sum().item()

        metrics["train/loss"] += clean_loss.item()
        metrics["train/accuracy"] += clean_correct / batch_size
        metrics["train/adv_loss"] += adv_loss.item()
        metrics["train/adv_accuracy"] += adv_correct / batch_size

    # Normalize metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


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
        # Train one epoch
        if config.adv.use_adversarial_training:
            metrics = train_one_epoch_adversarial(
                model=model,
                train_loader=train_loader,
                device=device,
                criterion=loss,
                optimizer=optimizer,
                epoch=epoch,
                epsilon=config.adv.epsilon,
                use_mixed_batch=True,
                mixed_ratio=0.4,
                dataset_mean=train_dataset.mean,
                dataset_std=train_dataset.std,
            )
        else:
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

            curvature = compute_curvature(
                model=model,
                dataloader=test_loader,
                device=device,
            )
            metrics.update({"curvature": curvature})

            # Adv test stats
            adv_metrics = evaluate_adversarial_robustness(
                model=model,
                dataloader=test_loader,
                device=device,
                dataset_mean=train_dataset.mean,
                dataset_std=train_dataset.std,
                epsilon=config.adv.epsilon,
            )
            metrics.update(adv_metrics)

            # Print out
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Loss: {metrics['train/loss']:.4f}, "
            )

        # Log all epoch metrics
        wandb.log(metrics, step=epoch)

    # Save model
    if config.save_dir:
        save_model(model=model, config=config, dataset_name=test_dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
