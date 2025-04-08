import json
import math
import os
import pdb
from ast import literal_eval
from collections import defaultdict
from copy import copy
from pathlib import Path

import click
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print as rprint
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchvision import transforms
from torchvision.utils import save_image

from analysis.common import load_autoencoder, load_model
from koopmann.data import (
    DatasetConfig,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.mixins.serializable import Serializable
from koopmann.models import BaseTorchModel
from koopmann.utils import get_device

############################
USER = os.environ.get("USER")


# Define attacks and epsilons
attacks = [
    fb.attacks.FGSM(),
    # fb.attacks.L2PGD(),
    fb.attacks.LinfPGD(),
    fb.attacks.L2CarliniWagnerAttack(),
    # fb.attacks.LinfBasicIterativeAttack(),
    # fb.attacks.LinfDeepFoolAttack(),
]
epsilons = [
    0.0,
    0.005,
    0.01,
    0.03,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.15,
]
############################


def load_dataset(metadata, data_path):
    dataset_config = DatasetConfig(
        dataset_name=metadata["dataset"],
        num_samples=5_000,
        split="test",
        seed=21,
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)

    return dataset


class FrankensteinCouncil(nn.Module):
    def __init__(self, frankenstein_models, original_model):
        super().__init__()
        self.frankenstein_models = frankenstein_models
        self.original_model = original_model

    def forward(self, x) -> list:
        res = []
        res.append(self.original_model(x))
        for fm in self.frankenstein_models:
            res.append(fm(x))

        return torch.stack(res)


class FrankensteinModel(nn.Module):
    def __init__(self, original_model, autoencoder, scale_idx=0):
        super().__init__()

        self.k_steps = autoencoder.k_steps
        self.scale_idx = scale_idx
        self.rank = autoencoder.rank
        self.latent_features = autoencoder.latent_features
        self.start_components = nn.Sequential(
            original_model.components[: self.scale_idx],
        )
        self.autoencoder = autoencoder
        self.end_components = nn.Sequential(original_model.components[-1:])

    def forward(self, x) -> torch.Tensor:
        x = self.start_components(x)
        x = self.autoencoder(x, k=self.k_steps).predictions[-1]
        x = self.end_components(x)
        return x


class NestedDefaultDict(defaultdict):
    def __init__(self):
        super().__init__(NestedDefaultDict)

    def __repr__(self):
        return dict(self).__repr__()


def plot_attack_results(attack_summary):
    # For each attack type
    for attack_name in attack_summary:
        # Get all scenarios for this attack
        scenarios = list(attack_summary[attack_name].keys())
        n_scenarios = len(scenarios)

        # Calculate optimal grid layout (max 3 columns)
        n_cols = min(n_scenarios, 3)
        n_rows = math.ceil(n_scenarios / n_cols)

        # Create a figure with subplots (with appropriate layout)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(f"{attack_name}", fontsize=16)

        # For each scenario
        for i, scenario in enumerate(scenarios):
            # Calculate row and column index for this scenario
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axs[row_idx, col_idx]

            # For each model in this scenario
            for model_name in attack_summary[attack_name][scenario]:
                # Get epsilon-accuracy pairs
                data = attack_summary[attack_name][scenario][model_name]
                epsilons = sorted(data.keys())
                accuracies = [data[eps] for eps in epsilons]

                # Plot this model's line
                ax.plot(epsilons, accuracies, marker="o", label=model_name)

            # Set x-axis to logarithmic scale
            # ax.set_xscale("symlog", linthresh=0.001)
            ax.set_xscale("linear")

            # Add labels and legend
            ax.set_xticks([eps for eps in epsilons])
            ax.set_xticklabels([str(eps) for eps in epsilons], rotation=60)
            ax.set_xticklabels([str(eps) for eps in epsilons])
            ax.set_xlabel("Epsilon")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Model attacked: {scenario}")
            if i == 0:
                ax.legend()
            # ax.grid(True)

        # Hide empty subplots if any
        for j in range(i + 1, n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            axs[row, col].set_visible(False)

        plt.tight_layout()
        # Save figure with unique name based on attack
        fig.savefig(f"../output/{attack_name}_results.png", dpi=300, bbox_inches="tight")


def adv_attack(
    dataset,
    model: BaseTorchModel,
    ae_names: list[str],
    autoencoders: list[BaseTorchModel],
    device: str,
):
    # Prepare data
    normalize_transform = transforms.Lambda(lambda x: (x / 255))
    images, labels = normalize_transform(dataset.data).to(device), dataset.labels.to(device)
    fb_preprocessing = dict(mean=[images.mean()], std=[images.std()], axis=-1)

    # Prepare models
    frankenstein_models_list = []
    models = {
        "original": fb.PyTorchModel(model, bounds=(0, 1), preprocessing=fb_preprocessing),
    }
    model.eval()
    for ae_name, ae in zip(ae_names, autoencoders):
        name = f"ae_dim{ae.latent_features}_rank{ae.rank}"

        fb_frankenstein_model = fb.PyTorchModel(
            FrankensteinModel(model, ae).eval(), bounds=(0, 1), preprocessing=fb_preprocessing
        )
        frankenstein_models_list.append(fb_frankenstein_model)
        models[name] = fb_frankenstein_model

    council = FrankensteinCouncil(frankenstein_models_list, model)
    models.update({"error_corrected": council})

    # Define testing scenarios
    scenarios = [
        "original",
        # "ae_dim256_rank20",
        # "ae_dim512_rank10",
        # "ae_dim512_rank20",
        # "ae_dim1024_rank10",
        "ae_dim1024_rank20",
        # "ae_dim2048_rank10",
        "ae_dim2048_rank20",
    ]
    attack_summary = NestedDefaultDict()

    for attack in attacks:
        attack_name = attack.__class__.__name__

        for scenario in scenarios:
            # Run the attack
            _, clipped_advs_per_epsilon, success = attack(
                models[scenario], images, labels, epsilons=epsilons
            )
            assert success.shape == (len(epsilons), len(images))

            for i, adv_tensor in enumerate(clipped_advs_per_epsilon):
                for model_name, eval_model in models.items():
                    # Accuracy of attacked
                    if model_name == scenario:
                        acc = ((len(images) - success[i].sum()) / len(images)).item()
                    # Accuracy of others
                    else:
                        metric = MulticlassAccuracy()
                        if isinstance(eval_model, FrankensteinCouncil):
                            output_list = eval_model(adv_tensor)  # [models, batch, features]
                            max_output_list = torch.argmax(output_list, dim=2)  # [models, batch]
                            output, _ = torch.mode(max_output_list, dim=0)  # [batch]
                        else:
                            output = eval_model(adv_tensor)  # [batch, features]
                        metric.update(output, labels.squeeze())
                        acc = metric.compute().item()

                    # Store the result
                    attack_summary[attack_name][scenario][model_name][epsilons[i]] = acc * 100
            logger.info(f"Scenario {scenario} done.")

    return attack_summary


def analyze_lipschitz_constants(model, n_samples=1000):
    # Generate random samples and small perturbations
    samples = torch.randn(n_samples, model.in_features).to(get_device())
    delta = 1e-4
    perturbed = samples + delta * torch.randn_like(samples)

    # Compute Jacobian norms (approximate Lipschitz constants)
    enc_lip = compute_empirical_lipschitz(lambda x: model.encode(x), samples, perturbed, delta)

    koop_lip = compute_empirical_lipschitz(
        lambda x: model.components.koopman_matrix(x),
        model.encode(samples),
        model.encode(perturbed),
        delta,
    )

    dec_lip = compute_empirical_lipschitz(
        lambda x: model.decode(x),
        model.components.koopman_matrix(model.encode(samples)),
        model.components.koopman_matrix(model.encode(perturbed)),
        delta,
    )

    return {"encoder": enc_lip, "koopman": koop_lip, "decoder": dec_lip}


def compute_empirical_lipschitz(function, x, x_perturbed, delta):
    with torch.no_grad():
        y = function(x)
        y_perturbed = function(x_perturbed)

    output_diff = torch.norm(y_perturbed - y, dim=1)
    input_diff = torch.norm(x_perturbed - x, dim=1)

    # Lipschitz constant is the maximum ratio of output difference to input difference
    lipschitz = (output_diff / (input_diff + 1e-10)).max().item()

    return lipschitz


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
    default=f"/scratch/{USER}/koopmann_model_saves/",
    # prompt=True,
    help="Location of MLP path.",
    type=click.Path(exists=True),
)
@click.option(
    "--ae_path",
    default=f"/scratch/{USER}/koopmann_model_saves/",
    # prompt=True,
    help="Autoencoder directory.",
    type=click.Path(exists=True),
)
def main(data_path, mlp_path, ae_path):
    device = get_device()

    # Load MLP
    model_name = "resmlp"
    model, model_metadata = load_model(mlp_path, model_name)
    model = model.to(device)

    # Load autoencoder
    ae_names = [
        # f"dim_{256}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model",
        # f"dim_{512}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        # f"dim_{512}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model",
        # f"dim_{1024}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        f"dim_{1024}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model_adv",
        # f"dim_{2048}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        f"dim_{2048}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model_adv",
    ]
    autoencoders = [load_autoencoder(ae_path, name)[0].to(device) for name in ae_names]

    # Load dataset
    dataset = load_dataset(model_metadata, data_path)

    # Sensitivity
    for name, ae in zip(ae_names, autoencoders):
        lipschitz_results = analyze_lipschitz_constants(ae)
        logger.info(name)
        logger.info(lipschitz_results)

    # Attack
    results = adv_attack(dataset, model, ae_names, autoencoders, device)

    # Writing to JSON file
    with open("../output/results_adv.json", "w+") as json_file:
        json.dump(results, json_file, indent=4)

    logger.info("Plotting.")
    # plot_attack_results(results)


if __name__ == "__main__":
    main()
