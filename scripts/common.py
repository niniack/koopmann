import pdb
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel

import wandb
from koopmann.log import logger
from koopmann.utils import set_seed

T = TypeVar("T", bound=BaseModel)


def setup_config(
    config_path_or_obj: Optional[Union[Path, str, T]] = None, config_type: Type[T] = None
) -> T:
    """
    Initializes configuration, sets up WandB, and ensures reproducibility.
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
    if not wandb_config_dict:
        cloned_config = deepcopy(dict(config))
        cloned_config = _convert_subdicts(cloned_config)
        wandb.config.update(cloned_config)

    set_seed(config.seed)

    return config


def load_config(config_path_or_obj: Path | str | T | dict, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.
    https://github.com/ApolloResearch/e2e_sae/blob/main/e2e_sae/utils.py

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
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


def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


def prepare_padded_acts_and_masks(act_dict, autoencoder):
    """
    Returns:
      padded_acts: list of padded activation tensors (one per layer)
      masks: list of 1/0 masks for ignoring "extra" neurons in padded activations
    """

    # Autoencoder input dimension
    ae_input_size = autoencoder.encoder[0].in_features
    padded_acts = []
    masks = []

    # Iterate through all activations
    for act in act_dict.values():
        # Pad activations
        padded_act = pad_act(act, ae_input_size)
        padded_acts.append(padded_act)

        # Construct a mask that has '1' up to original size, then 0
        mask = torch.zeros(ae_input_size, device=act.device)
        mask[: act.size(-1)] = 1
        masks.append(mask)

    return padded_acts, masks
