import os
import pdb
from pathlib import Path
from typing import Optional, Union

import fire
import torch
from config_def import Config
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
from torch import nn, optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import wandb
from koopmann.data import create_data_loader, get_dataset_class
from koopmann.log import logger
from koopmann.models import MLP
from koopmann.models.utils import get_device
from koopmann.utils import compute_model_accuracy
from scripts.common import setup_config


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


def train_one_epoch(
    model: Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: Module,
    optimizer: Optimizer,
) -> float:
    """Trains the model for one epoch and returns the average training loss.

    Args:
        model (Module): The PyTorch model to train.
        train_loader (DataLoader): The DataLoader providing training batches.
        device (torch.device): The device to run training on.
        criterion (Module): The loss function.
        optimizer (Optimizer): The optimizer for model updates.

    Returns:
        float: The average training loss for the epoch.
    """
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

    # Model setup and hooking
    model = MLP(
        config=config.model.hidden_neurons,
        input_dimension=dataset.in_features,
        output_dimension=config.model.out_features,
        nonlinearity="relu",
        bias=True,
    )
    model.to(device).train()
    model.hook_model()

    # Loss + optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )

    # Training loop
    for epoch in range(config.optim.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
        )

        with torch.no_grad():
            neural_collapse_metrics = compute_neural_collapse_metrics(
                model, config, train_loader, device
            )

        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss} | neural_collapse_metrics, step=epoch)

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

        model_path = Path(config.save_dir, f"{config.save_name}.safetensors")
        model.save_model(model_path, dataset=dataset.name())


if __name__ == "__main__":
    fire.Fire(main)
