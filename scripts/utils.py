import pdb
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Type, TypeVar, Union

import torch
import wandb
import yaml
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
from pydantic import BaseModel

from koopmann.log import logger
from koopmann.utils import set_seed

T = TypeVar("T", bound=BaseModel)


def separate_param_groups(model, weight_decay):
    """
    Separates parameters into groups with and without weight decay
    without relying on model.modules.
    """
    decay = []
    no_decay = []
    bn_params = set()

    # Recursive function to identify BatchNorm layers
    def find_bn_parameters(module):
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            bn_params.update(module.parameters())

        # Use children() method to iterate through direct children
        for child in module.children():
            find_bn_parameters(child)

    # Start recursive search from the model
    find_bn_parameters(model)

    # Categorize all parameters
    for name, param in model.named_parameters():
        if any(param is bn_param for bn_param in bn_params):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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


def compute_neural_collapse_metrics(model, config, train_loader, device):
    model.eval()
    # weights = model.fc.weight

    layer_key = list(model.get_fwd_activations().keys())[-2]
    num_classes = config.model.out_features
    d_vectors = config.model.hidden_neurons[-1]

    # Mean
    mean_accum = MeanAccumulator(
        n_classes=num_classes,
        d_vectors=d_vectors,
        device=device,
    )
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        _ = model(images)
        act_dict = model.get_fwd_activations()
        mean_accum.accumulate(act_dict[layer_key], labels)
    means, mG = mean_accum.compute()

    # Variance, covariance
    # var_norms_accum = VarNormAccumulator(
    #     n_classes=num_classes,
    #     d_vectors=d_vectors,
    #     device=device,
    # )
    # covar_accum = CovarAccumulator(
    #     n_classes=num_classes,
    #     d_vectors=d_vectors,
    #     device=device,
    #     M=means,
    # )
    # for images, labels in train_loader:
    #     images, labels = images.to(device), labels.to(device)
    #     _ = model(images)
    #     var_norms_accum.accumulate(act_dict[layer_key], labels, means)
    #     covar_accum.accumulate(act_dict[layer_key], labels, means)
    # var_norms, _ = var_norms_accum.compute()
    # covar_within = covar_accum.compute()

    # dec_accum = DecAccumulator(10, 512, "cuda", M=means, W=weights)
    # dec_accum.create_index(means)  # optionally use FAISS index for NCC
    # for i, (images, labels) in enumerate(test_loader):
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = model(images)

    #     # mean embeddings (only) necessary again if not using FAISS index
    #     if dec_accum.index is None:
    #         dec_accum.accumulate(Features.value, labels, weights, means)
    #     else:
    #         dec_accum.accumulate(Features.value, labels, weights)

    # ood_mean_accum = MeanAccumulator(10, 512, "cuda")
    # for i, (images, labels) in enumerate(ood_loader):
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = model(images)
    #     ood_mean_accum.accumulate(Features.value, labels)
    # _, mG_ood = ood_mean_accum.compute()

    # Neural collapse measurements
    nc_results_dict = {
        # "nc1_pinv": covariance_ratio(covar_within, means, mG),
        # "nc1_svd": covariance_ratio(covar_within, means, mG, "svd"),
        # "nc1_quot": covariance_ratio(covar_within, means, mG, "quotient"),
        # "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
        "nc2_etf_err": simplex_etf_error(means, mG),
        "nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
        "nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1],
        # "nc3_dual_err": self_duality_error(weights, means, mG),
        # "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
        # "nc4_agree": clf_ncc_agreement(dec_accum),
        # "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
    }

    return nc_results_dict
