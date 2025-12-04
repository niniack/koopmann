import os
from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel

from koopmann.models import DecompExponentialKoopmanAutencoder, KoopmanAutoencoder
from koopmann.utils import set_seed
from scripts.train_kae.config_def import KoopmanParam

TConfig = TypeVar("TConfig", bound=BaseModel)


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
    else:
        autoencoder = KoopmanAutoencoder(**autoencoder_kwargs).to(device)

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


def setup_config(
    config_path_or_obj: Path | str | dict | TConfig, config_model: Type[TConfig]
) -> TConfig:
    """
    Initializes configuration and sets up WandB
    """

    config = load_config(config_path_or_obj, config_model)

    if hasattr(config, "seed"):
        set_seed(config.seed)

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
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)

    return config_model(**config_dict)
