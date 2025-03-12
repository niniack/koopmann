import os
import pdb
from copy import deepcopy
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

from koopmann.data import create_data_loader, get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP, ResMLP
from koopmann.models.utils import get_device
from koopmann.utils import compute_model_accuracy
from scripts.utils import get_parameter_groups, setup_config


def train_one_epoch(
    model: Module,
    original_model: Module,
    train_loader: DataLoader,
    device: torch.device,
    ce_criterion: Module,
    mse_criterion: Module,
    optimizer: Optimizer,
) -> float:
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
        loss = ce_loss + 10 * mse_loss
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
    if "residual" in config.probe.model_to_probe:
        original_model, _ = ResMLP.load_model(file_path=config.probe.model_to_probe)
    else:
        original_model, _ = MLP.load_model(file_path=config.probe.model_to_probe)

    original_model.to(device).eval()

    # Clone model
    model = deepcopy(original_model)
    model.to(device).eval()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Remove a block
    insert_index = len(original_model.modules) - 2
    model.remove_layer(index=insert_index)

    # Insert layers
    model.insert_layer(index=insert_index, out_features=512, nonlinearity="relu")
    model.insert_layer(index=insert_index + 1, nonlinearity="relu")

    model.to(device)

    # Loss + optimizer
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    if config.optim.type.value == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.type.value == "sgd":
        # Use the helper function to create parameter groups
        param_groups = get_parameter_groups(model, config.optim.weight_decay)

        optimizer = optim.SGD(
            params=param_groups,
            momentum=0.9,
            lr=config.optim.learning_rate,
        )
    else:
        raise NotImplementedError("Pick either 'sgd' or 'adamw'")

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

        save_name = config.save_name
        if "residual" in config.probe.model_to_probe:
            save_name = config.save_name + "_residual"
        model_path = Path(config.save_dir, f"{save_name}.safetensors")
        model.save_model(model_path, dataset=dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
