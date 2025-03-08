import os
import pdb
from pathlib import Path
from typing import Optional, Union

import fire
import torch
import wandb
from config_def import Config
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from koopmann.data import get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP, ConvResNet, ResMLP
from koopmann.utils import compute_model_accuracy, get_device
from scripts.utils import separate_param_groups, setup_config


def get_dataloaders(config):
    # Train data
    DatasetClass = get_dataset_class(name=config.train_data.dataset_name)
    train_dataset = DatasetClass(config=config.train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
    )

    # Test data
    original_split = config.train_data.split
    config.train_data.split = "test"
    test_dataset = DatasetClass(config=config.train_data)
    config.train_data.split = original_split
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
    )

    return train_loader, test_loader, test_dataset


def get_model(config, dataset_features):
    # Model setup and hooking
    if config.model.residual:
        model = ResMLP(
            in_features=dataset_features,
            out_features=config.model.out_features,
            hidden_config=config.model.hidden_neurons,
            bias=False,
            batchnorm=True,
            nonlinearity="relu",
        )
    else:
        model = MLP(
            in_features=dataset_features,
            out_features=config.model.out_features,
            hidden_config=config.model.hidden_neurons,
            bias=False,
            batchnorm=True,
            nonlinearity="relu",
        )

    return model


def get_optimizer(config, model):
    # For both SGD and AdamW, we want to avoid weight decay on batch norm parameters
    param_groups = separate_param_groups(model, config.optim.weight_decay)

    # Optimizer
    if config.optim.type.value == "adamw":
        optimizer = optim.AdamW(
            params=param_groups,
            lr=config.optim.learning_rate,
        )
    elif config.optim.type.value == "sgd":
        optimizer = optim.SGD(
            params=param_groups,
            momentum=0.9,
            lr=config.optim.learning_rate,
        )
    else:
        raise NotImplementedError("Pick either 'sgd' or 'adamw'")

    return optimizer


def log_model_stats(model, step, log_histograms=False):
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

    # Log everything at once
    wandb.log(stats, step=step)


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
    epoch: int,
) -> float:
    model.to(device).train()
    loss_epoch = 0.0

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = (
            input.to(device, non_blocking=True),
            label.to(device, non_blocking=True).squeeze(),
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(input)
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
    model.to(device).train()

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
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Loss: {train_loss:.4f}, "
                f"Eval Acc: {test_acc:.4f}"
            )

        # Log all epoch metrics
        wandb.log(metrics, step=epoch)

        # Log histogram data (expensive operation) every 5 epochs
        if epoch % 5 == 0:
            log_model_stats(model, epoch, log_histograms=True)

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        model_path = Path(config.save_dir)
        model.save_model(model_path)


if __name__ == "__main__":
    fire.Fire(main)
