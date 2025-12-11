from pathlib import Path

import fire
import torch
import wandb
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from koopmann.log import logger
from koopmann.utils import get_device
from train.common_config_def import EnvSettings
from train.common_utils import (
    get_lr_schedule,
    get_model,
    get_optimizer,
    save_model,
    setup_config,
)
from train.mlp.config_def import Config
from train.mlp.utils import compute_model_stats, evaluate_model, get_dataloaders


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

    for _, (inputs, labels) in enumerate(train_loader):
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

    # Normalize metrics by number of batches
    metrics["train/loss"] /= num_batches
    metrics["train/accuracy"] /= num_batches
    metrics["epoch"] = epoch

    return metrics


def main(
    config_path_or_obj: Path | str | Config,
    env: EnvSettings = EnvSettings(),  # Load env vars by default # type: ignore
):
    # Config
    config = setup_config(config_path_or_obj, Config, env)
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
    if config.optim.scheduler is None:
        scheduler = None
    else:
        scheduler = get_lr_schedule(
            lr_schedule_type=config.optim.scheduler,
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

            eval_metrics = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
            )

            metrics.update(eval_metrics)

            # Print out
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs},"
                f"Loss: {metrics['train/loss']:.4f},"
                f"Test Accuracy: {metrics['test/accuracy']:.4f}"
            )

        # Log all epoch metrics
        if config.wandb.use_wandb and (epoch + 1) % config.print_freq // 2 == 0:
            wandb.log(metrics, step=epoch)

    # Save model
    save_model(
        model,
        env.WEIGHTS_CACHE,
        save_name=config.save_name,
        dataset_name=test_dataset.name(),
    )


def fire_main():
    torch.set_printoptions(precision=4)
    fire.Fire(main)


if __name__ == "__main__":
    fire_main()
