import os
from pathlib import Path
from typing import Any, TypeVar

import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from scripts.train_kae.config_def import KoopmanParam
from torch import optim

from koopmann.models import (
    KoopmanAutoencoder,
    ParamExponentialKoopmanAutencoder,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")
TConfig = TypeVar("TConfig", bound=BaseModel)
WEIGHTS_CACHE = os.getenv("WEIGHTS_CACHE")


############# OPTIMIZER #############
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


def separate_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []

    # Track parameters we've seen to avoid duplicates
    seen_params = set()

    # First, scan through the model's modules to categorize parameters
    for module_name, module in model.named_modules():
        # Skip the root module
        if module_name == "":
            continue

        # # Skip Koopman modules entirely
        # if "koopman" in module_name.lower():
        #     for param_name, param in module.named_parameters(recurse=False):
        #         full_name = f"{module_name}.{param_name}"
        #         if param.requires_grad and id(param) not in seen_params:
        #             no_decay_params.append(param)
        #             seen_params.add(id(param))
        #     continue

        # Skip BatchNorm modules entirely
        if isinstance(
            module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}"
                if param.requires_grad and id(param) not in seen_params:
                    no_decay_params.append(param)
                    seen_params.add(id(param))
            continue

        # For other modules, exclude biases
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}"
            if param.requires_grad and id(param) not in seen_params:
                if "bias" in param_name.lower():
                    no_decay_params.append(param)
                    seen_params.add(id(param))
                else:
                    decay_params.append(param)
                    seen_params.add(id(param))

    # Check for any parameters we missed (can happen with custom parameter registrations)
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in seen_params:
            if "bias" in name.lower() or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
            seen_params.add(id(param))

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


############# AUTOENCODER #############
def build_autoencoder(config: Any, device: str | torch.device) -> KoopmanAutoencoder:
    autoencoder_kwargs = {
        "k_steps": config.autoencoder.k_steps,
        "in_features": config.autoencoder.in_features,
        "latent_features": config.autoencoder.ae_dim,
        "hidden_config": config.autoencoder.hidden_config,
        "batchnorm": config.autoencoder.batchnorm,
        "bias": config.autoencoder.bias,
        "nonlinearity": config.autoencoder.ae_nonlinearity,
        "use_eigeninit": False,
    }

    if config.autoencoder.koopman_param == KoopmanParam.exponential:
        autoencoder = ParamExponentialKoopmanAutencoder(**autoencoder_kwargs).to(device)
    else:
        autoencoder = KoopmanAutoencoder(**autoencoder_kwargs).to(device)

    return autoencoder


def save_autoencoder(autoencoder, config, save_dir=None, **kwargs):
    if save_dir is None:
        save_dir = WEIGHTS_CACHE

    os.makedirs(save_dir, exist_ok=True)

    if hasattr(config, "suffix"):
        suffix = config.suffix + f"_seed_{config.seed}"
    else:
        suffix = f"_seed_{config.seed}"

    filename = (
        f"dim_{config.autoencoder.ae_dim}_"
        f"k_{config.autoencoder.k_steps}_"
        f"autoencoder_{config.save_name}"
        f"{suffix}"
        ".safetensors"
    )

    ae_path = Path(save_dir, filename)

    # NOTE: We manually apply suffix beforehand
    autoencoder.save_model(ae_path, suffix=None, **kwargs)

    return ae_path
