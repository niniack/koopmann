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

from koopmann.data import create_data_loader, get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP, ConvResNet, ResMLP
from koopmann.models.utils import get_device
from koopmann.utils import compute_model_accuracy
from scripts.common import get_parameter_groups, setup_config


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
) -> float:
    model.train()
    loss_epoch = 0.0

    for input, label in train_loader:
        input, label = input.to(device), label.to(device).squeeze()
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    return loss_epoch / len(train_loader)


def main(config_path_or_obj: Optional[Union[Path, str, Config]] = None):
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

    # Model setup and hooking
    if config.model.residual:
        model = ResMLP(
            input_dimension=dataset.in_features,
            output_dimension=config.model.out_features,
            config=config.model.hidden_neurons,
            nonlinearity="relu",
            bias=False,
            batchnorm=True,
            stochastic_depth_prob=0.3,
        )
    else:
        model = MLP(
            input_dimension=dataset.in_features,
            output_dimension=config.model.out_features,
            config=config.model.hidden_neurons,
            nonlinearity="relu",
            bias=False,
            batchnorm=True,
        )

    model.to(device).train()
    model.hook_model()

    # Loss + optimizer
    loss = nn.CrossEntropyLoss()

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.optim.num_epochs)
    # scheduler = None

    # Training loop
    for epoch in range(config.optim.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
        )

        scheduler.step()

        # with torch.no_grad():
        #     neural_collapse_metrics = compute_neural_collapse_metrics(
        #         model, config, train_loader, device
        #     )
        neural_collapse_metrics = {}

        if not scheduler:
            lr = config.optim.learning_rate
        else:
            lr = scheduler.get_last_lr()[0]

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": lr,
            }
            | neural_collapse_metrics,
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
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, " f"Loss: {train_loss:.4f}, "
            )

    # Save model
    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

        save_name = config.save_name
        if config.model.residual:
            save_name = config.save_name + "_residual"
        model_path = Path(config.save_dir, f"{save_name}.safetensors")
        model.save_model(model_path, dataset=dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
