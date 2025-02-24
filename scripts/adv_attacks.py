import os
import pdb
from ast import literal_eval
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

############################
USER = os.environ.get("USER")
task = "mnist"
scale_idx = "0"
k = "1"
dim = "1024"
use_exponential = True
############################


def load_MLP(mlp_path):
    model, _ = MLP.load_model(mlp_path)
    model.modules[-2].remove_nonlinearity()
    # model.modules[-3].update_nonlinearity("leakyrelu")
    model.modules[-3].remove_nonlinearity()
    model.eval().hook_model()
    return model


def load_autoencoder(ae_path, use_exponential):
    # Parse metadata
    metadata = parse_safetensors_metadata(file_path=ae_path)

    # Choose model based on flag
    AutoencoderClass = LowRankKoopmanAutoencoder if use_exponential else Autoencoder

    # Instantiate model
    autoencoder = AutoencoderClass(
        input_dimension=literal_eval(metadata["input_dimension"]),
        latent_dimension=literal_eval(metadata["latent_dimension"]),
        nonlinearity=metadata["nonlinearity"],
        k=literal_eval(metadata["steps"]),
        batchnorm=literal_eval(metadata["batchnorm"]),
        hidden_configuration=literal_eval(metadata["hidden_configuration"]),
    )

    # Load weights
    load_model(autoencoder, ae_path, device=get_device(), strict=True)
    autoencoder.eval()

    return autoencoder


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
    default=f"/scratch/{USER}/koopmann_model_saves/scaling",
    # prompt=True,
    help="Autoencoder directory.",
    type=click.Path(exists=True),
)
def main(data_path, mlp_path, ae_path):
    # Load MLP and Autoencoder
    model = load_MLP(mlp_path)
    ae_path = Path.joinpath(
        Path(ae_path), f"k_{k}_dim_{dim}_loc_{scale_idx}_autoencoder_{task}_model.safetensors"
    )
    autoencoder = load_autoencoder(ae_path, use_exponential)

    # Load dataset
    dataset = load_dataset(mlp_path, data_path)
    dataloader = create_data_loader(dataset, batch_size=1024)

    # Set up FoolBox
    normalize_transform = transforms.Lambda(lambda x: (x / (255 / 2)) - 1)
    images, labels = normalize_transform(dataset.data), dataset.labels
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
    attack_summary = {attack.__class__.__name__: {"naive": {}, "koopman": {}} for attack in attacks}
    epsilons = [
        0.0,
        0.001,
        0.003,
        0.005,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    ]

    # Loop through attacks
    # attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=bool)

    for attack in attacks:
        attack_name = attack.__class__.__name__
        _, clipped_adv, success = attack(fmodel, images, labels, epsilons=epsilons)
        assert success.shape == (len(epsilons), total_images)
        num_successful_images = torch.sum((success == torch.ones_like(success)), axis=-1)
        attack_summary[attack_name]["naive"] = 1 - (num_successful_images / total_images)

        epsilon_tensor = torch.empty(size=(len(epsilons),))
        for i, adv_tensor in enumerate(clipped_adv):
            koopman_metric = MulticlassAccuracy()
            autencoder_preds = autoencoder(adv_tensor, k=int(k)).predictions[-1]
            mlp_koopman_output = model.modules[-2:](autencoder_preds)
            koopman_metric.update(mlp_koopman_output, labels.squeeze())
            epsilon_tensor[i] = koopman_metric.compute()

        attack_summary[attack_name]["koopman"] = epsilon_tensor

    rprint(attack_summary)


if __name__ == "__main__":
    main()
