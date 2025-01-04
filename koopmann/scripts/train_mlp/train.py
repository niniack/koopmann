import os
import pdb
import sys
from pathlib import Path
from typing import Optional

import fire
import torch
import wandb
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from torch import nn, optim
from torch.utils.data import DataLoader

from koopmann.data import DatasetConfig, create_data_loader, get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP, BaseTorchModel
from koopmann.models.utils import get_device
from koopmann.scripts.common import load_config
from koopmann.utils import set_seed


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    out_features: PositiveInt
    hidden_neurons: list[PositiveInt]


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    weight_decay: NonNegativeFloat
    num_epochs: PositiveInt | None = None
    learning_rate: PositiveFloat


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    model: ModelConfig
    seed: NonNegativeInt = 0
    print_freq: PositiveInt
    batch_size: PositiveInt
    save_name: Optional[str] = None
    save_dir: Optional[str] = None


def train_one_epoch(
    model: BaseTorchModel,
    train_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn,
    optimizer: torch.nn,
) -> None:
    model.train()
    loss_epoch = 0

    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        label = label.squeeze()

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()  # Accumulate loss

    epoch_loss = loss_epoch / len(train_loader)  # Average loss over all batches
    return epoch_loss


def setup(config_path_or_obj: Optional[Path | str | Config] = None):
    # Initialize wandb
    wandb.init(entity="nishantaswani", project="scaling")

    # Validate
    if config_path_or_obj is None and not dict(wandb.config):
        sys.exit("No configuration found for the run! Provide a file")

    # Update wandb config
    wandb_config_dict = dict(wandb.config)
    if wandb_config_dict:
        wandb_config_dict.update({"save_dir": None, "print_freq": 500})

    config = load_config(
        config_path_or_obj or wandb_config_dict,
        config_model=Config,
    )
    logger.info(config)

    set_seed(config.seed)

    return config


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Setup WandB and load config
    config = setup(config_path_or_obj)

    # Device
    device = get_device()

    # Set up dataset
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset,
        batch_size=config.batch_size,
        global_seed=config.seed,
    )

    # Load model
    model = MLP(
        config=config.model.hidden_neurons,
        input_dimension=dataset.in_features,
        output_dimension=config.model.out_features,
        nonlinearity="relu",
        bias=True,
    )
    model.to(device).train()

    # Define loss and optimiser
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )

    # Loop over epochs
    for epoch in range(config.optim.num_epochs):
        # Train step
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
        )

        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(f"Epoch {epoch + 1}/{config.optim.num_epochs}, Loss: {train_loss:.4f}")

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        path = Path(config.save_dir, f"{config.save_name}.safetensors")
        model.save_model(path, dataset=dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
