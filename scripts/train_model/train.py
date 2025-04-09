import os
import pdb
from pathlib import Path
from typing import Optional, Union

import fire
import torch
from config_def import Config
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
from koopmann.log import logger
from koopmann.models import MLP, ConvResNet, ResMLP
from koopmann.utils import compute_model_accuracy, get_device
from scripts.utils import (
    compute_neural_collapse_metrics,
    denorm,
    fgsm_attack,
    get_dataloaders,
    get_optimizer,
    separate_param_groups,
    setup_config,
)

# NOTE: for now
ADV_EPSILON = 0.08
ADV_FRACTION = 0.5


def get_model(config, dataset_features):
    # Model setup and hooking
    if config.model.residual:
        model = ResMLP(
            in_features=dataset_features,
            out_features=config.model.out_features,
            hidden_config=config.model.hidden_neurons,
            bias=config.model.bias,
            batchnorm=config.model.batchnorm,
            nonlinearity="relu",
        )
    else:
        model = MLP(
            in_features=dataset_features,
            out_features=config.model.out_features,
            hidden_config=config.model.hidden_neurons,
            bias=config.model.bias,
            batchnorm=config.model.batchnorm,
            nonlinearity="relu",
        )

    return model


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


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
    epoch: int,
    use_adversarial_training: bool,
) -> float:
    model.to(device).train()
    loss_epoch = 0.0

    for _, (input, label) in enumerate(train_loader):
        # Unwrap input, label
        input, label = (
            input.to(device, non_blocking=True),
            label.to(device, non_blocking=True).squeeze(),
        )
        batch_size = input.size(0)
        input_clean = input.clone()

        if use_adversarial_training:
            # Set gradients for input
            input.requires_grad_()

            # Forward and backward pass
            output = model(input)
            loss = criterion(output, label.long())
            loss.backward()

            # Get gradients for input
            input_grad = input.grad.data

            # Denormalize
            input_denorm = denorm(input, device, mean=[0.1307], std=[0.3081])

            # Attack
            adv_input = fgsm_attack(input_denorm, ADV_EPSILON, input_grad)

            # Renormalize
            adv_input_normalized = transforms.Normalize((0.1307,), (0.3081,))(adv_input)

            # Replace random samples
            num_to_replace = int(ADV_FRACTION * batch_size)
            rand_idx = torch.randperm(batch_size)[:num_to_replace]
            input_clean[rand_idx] = adv_input_normalized[rand_idx]

        # Forward pass
        output = model(input_clean)

        optimizer.zero_grad()
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch / len(train_loader)


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
    # Config
    config = setup_config(config_path_or_obj, Config)
    device = get_device()

    # Data
    train_loader, test_loader, test_dataset = get_dataloaders(config)

    # Model
    model = get_model(config, test_dataset.in_features)
    model.to(device).train().hook_model()

    # Loss
    loss = nn.CrossEntropyLoss()

    # Optim
    optimizer = get_optimizer(config, model)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim.num_epochs)

    for epoch in range(config.optim.num_epochs):
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
            epoch=epoch,
            use_adversarial_training=config.adversarial_training,
        )

        # Log metrics
        if scheduler:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = config.optim.learning_rate

        # Log epoch-level metrics
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "lr": lr,
        }

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            test_acc = compute_model_accuracy(model, test_loader, device)
            metrics["test/accuracy"] = test_acc
            model_stats = compute_model_stats(model, epoch, log_histograms=True)
            nc_stats = compute_neural_collapse_metrics(model, config, test_loader, device)
            metrics.update(model_stats)
            metrics.update(nc_stats)
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Loss: {train_loss:.4f}, "
                f"Eval Acc: {test_acc:.4f}"
            )

        # Log all epoch metrics
        wandb.log(metrics, step=epoch)

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        model_path = Path(config.save_dir)
        suffix = "adv" if config.adversarial_training else None
        model.save_model(model_path, suffix=suffix, dataset=test_dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
