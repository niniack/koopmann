"""
Shared utility functions for training and evaluation.
"""

import os
from pathlib import Path
from typing import Type, TypeVar

import numpy as np
import yaml
from pydantic import BaseModel
from torch import nn, optim
from torch.optim.lr_scheduler import CyclicLR

import wandb
from koopmann.models import ResMLP
from koopmann.utils import set_seed
from train.common_config_def import EnvSettings

TConfig = TypeVar("TConfig", bound=BaseModel)


# Optimization
def get_optimizer(config, model):
    param_groups = separate_param_groups(model, config.optim.weight_decay)

    opt_type = config.optim.type.value.lower()
    if opt_type == "adamw":
        return optim.AdamW(
            params=param_groups,
            lr=config.optim.learning_rate,
        )
    elif opt_type == "sgd":
        return optim.SGD(
            params=param_groups,
            lr=config.optim.learning_rate,
            momentum=0.9,
        )
    else:
        raise NotImplementedError("Pick either 'sgd' or 'adamw'")


def separate_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []

    # map from module full name -> module
    modules = dict(model.named_modules())

    for full_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = modules[module_name]
        else:
            module_name, param_name = "", full_name
            module = model

        # no weight decay on BatchNorm parameters
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            no_decay_params.append(param)
        # no weight decay on biases
        elif "bias" in param_name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max, optimizer):
    lr_schedule_type = lr_schedule_type.value.lower()
    if lr_schedule_type == "cyclic":
        # NOTE: To avoid division by zero when debugging
        if n_epochs == 0:
            n_epochs = 100

        step_size_up = int(n_epochs * 1 / 5)
        step_size_down = n_epochs - step_size_up
        scheduler = CyclicLR(
            optimizer,
            base_lr=0,
            max_lr=lr_max,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False,  # Don't adjust momentum
            mode="triangular",  # Use triangular pattern (linear up, linear down)
        )
    else:
        raise ValueError("wrong lr_schedule_type")
    return scheduler


# Model
def get_model(config, dataset):
    """Get and load models."""
    model = ResMLP(
        in_features=np.prod(dataset.in_features),
        out_features=dataset.out_features,
        hidden_config=config.model.hidden_neurons,
        bias=config.model.bias,
        batchnorm=config.model.batchnorm,
        nonlinearity="relu",
    )
    return model


def save_model(model, save_dir, save_name, **kwargs):
    """Save models after training."""

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    model_path = Path(save_dir)

    # kwargs are metadata about the model
    model.save_model(model_path, suffix=save_name, **kwargs)


# Pydantic Config
def setup_config(
    config_path_or_obj: Path | str | dict | TConfig,
    config_model: Type[TConfig],
    env: EnvSettings,
) -> TConfig:
    """
    Initializes configuration and sets up WandB
    """

    # Parse config string
    config = load_config(config_path_or_obj, config_model)

    # Set seed
    if hasattr(config, "seed"):
        set_seed(config.seed)  # type: ignore

    # Init wandb
    if hasattr(config, "wandb") and config.wandb.use_wandb:  # type: ignore
        wandb.login(key=env.WANDB_API_KEY)
        wandb.init(
            entity=env.WANDB_ENTITY,
            project=env.WANDB_PROJECT,
            config=config.model_dump(),
            reinit=True,
        )

    return config


def load_config(
    config_path_or_obj: Path | str | dict | TConfig, config_model: Type[TConfig]
) -> TConfig:
    """Load the config of class `config_model`, either from YAML file or existing config object.
    https://github.com/ApolloResearch/e2e_sae/blob/main/e2e_sae/utils.py
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, dict):
        return config_model(**config_path_or_obj)

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(
        config_path_or_obj
    ).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)

    return config_model(**config_dict)
