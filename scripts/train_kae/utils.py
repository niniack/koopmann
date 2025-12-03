import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import wandb
import yaml
from pydantic import BaseModel

from koopmann.log import logger
from koopmann.models import DecompExponentialKoopmanAutencoder, KoopmanAutoencoder
from koopmann.utils import set_seed
from scripts.train_kae.config_def import KoopmanParam


############# AUTOENCODER #############
def build_autoencoder(config, device):
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
        autoencoder = DecompExponentialKoopmanAutencoder(**autoencoder_kwargs).to(device)
        flavor = config.autoencoder.koopman_param.value
        flavor = f"{config.autoencoder.koopman_param.value}_{config.autoencoder.koopman_rank}"
    else:
        autoencoder = KoopmanAutoencoder(**autoencoder_kwargs).to(device)
        flavor = "standard"

    return autoencoder


def save_autoencoder(autoencoder, config, flavor, **kwargs):
    if not config.save_dir:
        return None

    os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

    suffix = config.suffix if config.suffix else ""
    suffix = suffix + f"_seed_{config.seed}"

    filename = (
        f"dim_{config.autoencoder.ae_dim}_"
        f"k_{config.autoencoder.k_steps}_"
        f"{flavor}_"
        f"autoencoder_{config.save_name}"
        f"{suffix}"
        ".safetensors"
    )

    ae_path = Path(config.save_dir, filename)

    # NOTE: We manually apply suffix beforehand
    autoencoder.save_model(ae_path, suffix=None, **kwargs)

    return ae_path


############# CONFIG #############
def setup_config(config_path_or_obj: Optional[Union[Path, str]] = None, config_type=None):
    """
    Initializes configuration and sets up WandB
    """

    # If no external config is provided, initialize WandB to fetch its config.
    if config_path_or_obj is None:
        wandb.init()  # Initialize with defaults (or add your desired parameters)
        wandb_config_dict = dict(wandb.config)
    else:
        # If you have an external config file or object, ignore wandb.config for now.
        wandb_config_dict = {}

    # Load your configuration from either the provided file/object or wandb's config.
    config = load_config(config_path_or_obj or wandb_config_dict, config_model=config_type)

    # Now that you have loaded your config, if WandB is enabled in your config, ensure it's properly initialized.
    if config.wandb.use_wandb:
        # If WandB wasn’t already initialized (because a file was provided) then initialize it now.
        if wandb.run is None:
            # Check that required wandb fields are present
            if not config.wandb.entity or not config.wandb.project:
                raise ValueError("You must provide a WandB entity and project name.")
            wandb.init(entity=config.wandb.entity, project=config.wandb.project)

    if config_type is None:
        raise ValueError("config_type must be provided to specify the configuration class.")

    # If no configuration was provided via file and WandB’s config was empty, exit.
    if config_path_or_obj is None and not wandb_config_dict:
        sys.exit("No configuration found for the run! Please provide a file.")

    logger.info(config)

    def _convert_subdicts(config_dict):
        for key, val in config_dict.items():
            if isinstance(val, BaseModel):
                config_dict[key] = dict(val)
                _convert_subdicts(config_dict[key])
        return config_dict

    # Sync config to WandB if it was empty (only applicable if WandB was used)
    if config.wandb.use_wandb and not wandb_config_dict:
        cloned_config = deepcopy(dict(config))
        cloned_config = _convert_subdicts(cloned_config)
        wandb.config.update(cloned_config)

    set_seed(config.seed)

    return config


def load_config(config_path_or_obj: Path | str | dict, config_model):
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
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)

    return config_model(**config_dict)
