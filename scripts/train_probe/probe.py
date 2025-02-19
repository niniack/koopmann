import os
import pdb
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import fire
import torch
from config_def import Config
from neural_collapse.accumulate import (
    CovarAccumulator,
    DecAccumulator,
    MeanAccumulator,
    VarNormAccumulator,
)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (
    clf_ncc_agreement,
    covariance_ratio,
    orthogonality_deviation,
    self_duality_error,
    similarities,
    simplex_etf_error,
    variability_cdnv,
)
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import wandb
from koopmann.data import create_data_loader, get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP
from koopmann.models.utils import get_device
from koopmann.utils import compute_model_accuracy
from scripts.common import setup_config


def train_one_epoch(
    model: Module,
    original_model: Module,
    train_loader: DataLoader,
    device: torch.device,
    ce_criterion: Module,
    mse_criterion: Module,
    optimizer: Optimizer,
) -> float:
    """Trains the model for one epoch and returns the average training loss.

    Args:
        model (Module): The PyTorch model to train.
        train_loader (DataLoader): The DataLoader providing training batches.
        device (torch.device): The device to run training on.
        criterion (Module): The loss function.
        optimizer (Optimizer): The optimizer for model updates.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    ce_epoch = 0.0
    mse_epoch = 0.0

    for input, label in train_loader:
        input, label = input.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        output = model(input)
        with torch.no_grad():
            original_output = original_model(input)

        ce_loss = ce_criterion(output, label.long())
        mse_loss = mse_criterion(output, original_output)
        loss = ce_loss + mse_loss
        loss.backward()

        optimizer.step()
        ce_epoch += ce_loss.item()
        mse_epoch += mse_loss.item()

    return {
        "ce_loss": ce_epoch / len(train_loader),
        "mse_loss": mse_epoch / len(train_loader),
    }


def main(config_path_or_obj: Optional[Union[Path, str, Config]]):
    """Main training function that sets up the dataset, model, optimizer, and runs training.

    :param config_path_or_obj: Path to a configuration file, a config object, or None to use WandB's config.
    :type config_path_or_obj: Optional[Union[Path, str, Config]]
    """
    config = setup_config(config_path_or_obj, Config)
    device = get_device()

    # Data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset,
        batch_size=config.batch_size,
        global_seed=config.seed,
    )

    # Test data
    original_split = dataset_config.split
    dataset_config.split = "test"
    test_dataset = DatasetClass(config=dataset_config)
    dataset_config.split = original_split

    # Load model
    original_model, _ = MLP.load_model(file_path=config.probe.model_to_probe)
    original_model.to(device).eval()

    # Clone model
    model = deepcopy(original_model)
    model.to(device).eval()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Insert layers
    insert_index = len(original_model.modules) - 1
    model.insert_layer(index=insert_index, out_features=784, nonlinearity="leakyrelu")
    model.insert_layer(index=insert_index + 1, nonlinearity="none")
    model.to(device)

    # Loss + optimizer
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )

    # Training loop
    for epoch in range(config.optim.num_epochs):
        loss_dict = train_one_epoch(
            original_model=original_model,
            model=model,
            train_loader=train_loader,
            device=device,
            ce_criterion=ce_loss,
            mse_criterion=mse_loss,
            optimizer=optimizer,
        )

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "ce_loss": loss_dict["ce_loss"],
                "mse_loss": loss_dict["mse_loss"],
            },
            step=epoch,
        )

        # Evaluate
        if (epoch + 1) % config.print_freq == 0:
            test_acc = compute_model_accuracy(model, test_dataset)
            logger.info(f"Eval Acc: {test_acc}")
            wandb.log({"test_acc": test_acc}, step=epoch)

        # Print loss at specified intervals
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"CE Loss: {loss_dict['ce_loss']:.4f}, "
                f"MSE Loss: {loss_dict['mse_loss']:.4f}"
            )

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

        model_path = Path(config.save_dir, f"{config.save_name}.safetensors")
        model.save_model(model_path, dataset=dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
