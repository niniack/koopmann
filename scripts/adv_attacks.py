import os
import pdb
from ast import literal_eval
from copy import copy
from pathlib import Path

import click
import foolbox as fb
import numpy as np
import torch
from rich import print as rprint
from safetensors.torch import load_model
from torcheval.metrics import MulticlassAccuracy
from torchvision import transforms

from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
)
from koopmann.models.utils import get_device, parse_safetensors_metadata
from scripts.train_ae.config_def import KoopmanParam

############################
USER = os.environ.get("USER")
############################


def load_MLP(mlp_path):
    model, metadata = MLP.load_model(mlp_path)
    model.modules[-2].remove_nonlinearity()
    # model.modules[-3].update_nonlinearity("leaky_relu")
    model.modules[-3].remove_nonlinearity()
    model.eval().hook_model()
    return model, metadata


def load_autoencoder(ae_path):
    # Parse metadata
    metadata = parse_safetensors_metadata(file_path=ae_path)

    # Choose model based on flag
    if KoopmanParam.exponential.value in ae_path:
        AutoencoderClass = ExponentialKoopmanAutencoder
    elif KoopmanParam.lowrank.value in ae_path:
        AutoencoderClass = LowRankKoopmanAutoencoder
    else:
        AutoencoderClass = Autoencoder

    # Instantiate model
    autoencoder = AutoencoderClass(
        input_dimension=literal_eval(metadata["input_dimension"]),
        latent_dimension=literal_eval(metadata["latent_dimension"]),
        nonlinearity=metadata["nonlinearity"],
        k=literal_eval(metadata["steps"]),
        batchnorm=literal_eval(metadata["batchnorm"]),
        hidden_configuration=literal_eval(metadata["hidden_configuration"]),
        rank=literal_eval(metadata["rank"]),
    )

    # Load weights
    load_model(autoencoder, ae_path, device=get_device(), strict=True)
    autoencoder.eval()

    return autoencoder, metadata


def load_dataset(mlp_path, data_path):
    metadata = parse_safetensors_metadata(file_path=mlp_path)

    dataset_config = DatasetConfig(
        dataset_name=metadata["dataset"],
        num_samples=5_000,
        split="test",
        seed=21,
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)

    return dataset


@click.command()
@click.option(
    "--data_path",
    default=f"/scratch/{USER}/datasets",
    # prompt=True,
    help="Data directory.",
    type=click.Path(exists=True),
)
@click.option(
    "--mlp_path",
    default=f"/scratch/{USER}/koopmann_model_saves/mnist_probed.safetensors",
    # prompt=True,
    help="Location of MLP path.",
    type=click.Path(exists=True),
)
@click.option(
    "--ae_path",
    default=f"/scratch/{USER}/koopmann_model_saves/scaling/dim_1024_k_1_loc_0_lowrank_10_autoencoder_mnist_model.safetensors",
    # prompt=True,
    help="Autoencoder directory.",
    type=click.Path(exists=True),
)
def main(data_path, mlp_path, ae_path):
    device = get_device()

    # Load MLP and Autoencoder
    model, model_metadata = load_MLP(mlp_path)
    model = model.to(device)

    autoencoder, autoencoder_metadata = load_autoencoder(ae_path)
    autoencoder = autoencoder.to(device)
    k = int(autoencoder_metadata["steps"])

    # Load dataset
    dataset = load_dataset(mlp_path, data_path)
    dataloader = create_data_loader(dataset, batch_size=1024)

    # Set up FoolBox
    normalize_transform = transforms.Lambda(lambda x: (x / (255 / 2)) - 1)
    images, labels = normalize_transform(dataset.data).to(device), dataset.labels.to(device)
    total_images = len(images)
    preprocessing = dict(mean=[images.mean()], std=[images.std()], axis=-1)
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=preprocessing)

    # Set up FoolBox attacks
    attacks = [
        fb.attacks.FGSM(),
        fb.attacks.LinfPGD(),
        # fb.attacks.LinfBasicIterativeAttack(),
        # fb.attacks.LinfAdditiveUniformNoiseAttack(),
        # fb.attacks.LinfDeepFoolAttack(),
    ]
    epsilons = [
        0.0,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    ]

    # Set up results
    epsilon_summary = {epsilon: 0.0 for epsilon in epsilons}
    attack_summary = {
        attack.__class__.__name__: {
            "naive": copy(epsilon_summary),
            "koopman": copy(epsilon_summary),
        }
        for attack in attacks
    }

    # Loop through attacks
    for attack in attacks:
        # Get attack name
        attack_name = attack.__class__.__name__

        # Run attack on MLP
        _, clipped_adv, success = attack(fmodel, images, labels, epsilons=epsilons)
        assert success.shape == (len(epsilons), total_images)

        # Compute per epsilon rate
        num_successful_images = torch.sum((success == torch.ones_like(success)), axis=-1)
        mlp_epsilon_rate = 1 - (num_successful_images / total_images)

        # Run all epsilons on Koopman autoencoder
        koopman_epsilon_rate = torch.empty(size=(len(epsilons),))
        for i, adv_tensor in enumerate(clipped_adv):
            koopman_metric = MulticlassAccuracy()
            autencoder_preds = autoencoder(adv_tensor, k=int(k)).predictions[-1]
            koopman_output = model.modules[-2:](autencoder_preds)
            koopman_metric.update(koopman_output, labels.squeeze())
            koopman_epsilon_rate[i] = koopman_metric.compute()

        # Store
        for i, epsilon in enumerate(epsilons):
            attack_summary[attack_name]["naive"][epsilon] = mlp_epsilon_rate[i].item()
            attack_summary[attack_name]["koopman"][epsilon] = koopman_epsilon_rate[i].item()

    return attack_summary


if __name__ == "__main__":
    main()
