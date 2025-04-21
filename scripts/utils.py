import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar, Union

import torch
import yaml
from hessian_eigenthings import compute_hessian_eigenthings
from neural_collapse.accumulate import (
    CovarAccumulator,
    MeanAccumulator,
    VarNormAccumulator,
)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (
    covariance_ratio,
    simplex_etf_error,
)
from pydantic import BaseModel
from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader

import wandb
from koopmann.data import (
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.utils import set_seed

T = TypeVar("T", bound=BaseModel)


def compute_curvature(model, dataloader, device):
    eigenvalues, eigenvectors = compute_hessian_eigenthings(
        model=model,
        dataloader=dataloader,
        loss=torch.nn.CrossEntropyLoss(),
        num_eigenthings=1,
        full_dataset=True,
        mode="power_iter",
        power_iter_steps=20,
        power_iter_err_threshold=1e-3,
        use_gpu=True if device != "cpu" else False,
    )
    return eigenvalues[0]


def get_lr_schedule(lr_schedule_type: Literal["cyclic", "piecewise"], n_epochs, lr_max, optimizer):
    if lr_schedule_type == "cyclic":
        step_size_up = int(n_epochs * 2 / 5)
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


def get_dataloaders(config, train_batch_size=None, test_batch_size=None, shuffle=True):
    # Train data
    train_size = train_batch_size if train_batch_size else config.batch_size
    DatasetClass = get_dataset_class(name=config.train_data.dataset_name)
    train_dataset = DatasetClass(config=config.train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_size,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        prefetch_factor=4,
    )

    # Test data
    test_size = test_batch_size if test_batch_size else config.batch_size
    original_split = config.train_data.split
    config.train_data.split = "test"
    test_dataset = DatasetClass(config=config.train_data)
    config.train_data.split = original_split
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_size,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=True,
        num_workers=4,
        prefetch_factor=4,
    )

    return train_loader, test_loader, train_dataset, test_dataset


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


# Iterating function
def iterate_by_batches(act_dict, batch_size):
    keys = list(act_dict.keys())
    num_samples = act_dict[keys[0]].shape[0]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        # Create a dictionary with the current batch from each tensor
        batch_dict = OrderedDict((key, act_dict[key][start_idx:end_idx]) for key in keys)

        yield batch_dict


def compute_neural_collapse_metrics(model, config, dataloader, device):
    model.eval()
    layer_key = list(model.get_forward_activations().keys())[-2]
    num_classes = config.model.out_features
    d_vectors = config.model.hidden_neurons[-1]

    # Mean
    mean_accum = MeanAccumulator(
        n_classes=num_classes,
        d_vectors=d_vectors,
        device=device,
    )
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        _ = model(inputs)
        act_dict = model.get_forward_activations()
        mean_accum.accumulate(act_dict[layer_key], labels)
    means, mG = mean_accum.compute()

    # Variance, covariance
    var_norms_accum = VarNormAccumulator(
        n_classes=num_classes,
        d_vectors=d_vectors,
        device=device,
    )
    covar_accum = CovarAccumulator(
        n_classes=num_classes,
        d_vectors=d_vectors,
        device=device,
        M=means,
    )

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).squeeze()
        _ = model(inputs)
        act_dict = model.get_forward_activations()
        var_norms_accum.accumulate(act_dict[layer_key], labels, means)
        covar_accum.accumulate(act_dict[layer_key], labels, means)
    var_norms, _ = var_norms_accum.compute()
    covar_within = covar_accum.compute()

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
        "neural_collapse/nc1_pinv": covariance_ratio(covar_within, means, mG),
        "neural_collapse/nc1_svd": covariance_ratio(covar_within, means, mG, "svd"),
        "neural_collapse/nc1_quot": covariance_ratio(covar_within, means, mG, "quotient"),
        # "nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
        "neural_collapse/nc2_etf_err": simplex_etf_error(means, mG),
        "neural_collapse/nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
        "neural_collapse/nc2g_log": kernel_stats(means, mG, kernel=log_kernel, tile_size=64)[1],
        # "nc3_dual_err": self_duality_error(weights, means, mG),
        # "nc3u_uni_dual": similarities(weights, means, mG).var().item(),
        # "nc4_agree": clf_ncc_agreement(dec_accum),
        # "nc5_ood_dev": orthogonality_deviation(means, mG_ood),
    }

    return nc_results_dict


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
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
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


class DotDict(dict):
    """Dictionary subclass that provides attribute access to keys."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
