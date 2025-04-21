import json
import math
import os
from collections import defaultdict

import click
import foolbox as fb
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torcheval.metrics import MulticlassAccuracy
from torchvision import transforms

from analysis.common import load_autoencoder, load_model
from koopmann.data import (
    DatasetConfig,
    get_dataset_class,
)
from koopmann.log import logger
from koopmann.models import BaseTorchModel
from koopmann.utils import get_device

############################
USER = os.environ.get("USER")

# Define attacks and epsilons
attacks = [
    fb.attacks.FGSM(),
    # fb.attacks.L2PGD(),
    fb.attacks.LinfPGD(steps=50),
    # fb.attacks.L2CarliniWagnerAttack(),
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
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False, sharey=True
        )
        fig.suptitle(f"{attack_name}", fontsize=16)

        # Find global y-axis limits across all scenarios
        global_y_min = float("inf")
        global_y_max = float("-inf")

        # First pass to determine global y-axis limits
        for scenario in scenarios:
            for model_name in attack_summary[attack_name][scenario]:
                data = attack_summary[attack_name][scenario][model_name]
                accuracies = [data[eps] for eps in sorted(data.keys())]
                global_y_min = min(global_y_min, min(accuracies))
                global_y_max = max(global_y_max, max(accuracies))

        # Add a small buffer to the y-axis limits
        y_buffer = (global_y_max - global_y_min) * 0.05
        global_y_min = max(0, global_y_min - y_buffer)
        global_y_max = min(100, global_y_max + y_buffer)

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

            # Set y-axis limits consistently across all subplots
            ax.set_ylim(global_y_min, global_y_max)

            # Add labels and legend
            ax.set_xticks([eps for eps in epsilons])
            ax.set_xticklabels([str(eps) for eps in epsilons], rotation=60)
            ax.set_xlabel("Epsilon")

            # Only add y-label to plots in the first column
            if col_idx == 0:
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
        fig.savefig(f"../adv_output/{attack_name}_results.png", dpi=300, bbox_inches="tight")


def adv_attack(
    dataset: Dataset,
    model_names: list[str],
    ae_names: list[str],
    models: list[BaseTorchModel],
    autoencoders: list[BaseTorchModel],
    device: str,
):
    # Prepare data
    normalize_transform = transforms.Lambda(lambda x: (x / 255))
    images, labels = normalize_transform(dataset.data).to(device), dataset.labels.to(device)
    fb_preprocessing = dict(mean=[images.mean()], std=[images.std()], axis=-1)

    # Container for all models
    all_models_dict = {}

    # Prepare models
    for model_name, model in zip(model_names, models):
        all_models_dict[model_name] = fb.PyTorchModel(
            model.eval(), bounds=(0, 1), preprocessing=fb_preprocessing
        )

    # Prepare autoencoders
    frankenstein_models_list = []
    for ae_name, ae in zip(ae_names, autoencoders):
        new_ae_name = f"ae_dim{ae.latent_features}_rank{ae.rank}"

        fb_frankenstein_model = fb.PyTorchModel(
            FrankensteinModel(model, ae).eval(), bounds=(0, 1), preprocessing=fb_preprocessing
        )
        frankenstein_models_list.append(fb_frankenstein_model)
        all_models_dict[new_ae_name] = fb_frankenstein_model

    # # Prepare error correcting
    # council = FrankensteinCouncil(frankenstein_models_list, model)
    # all_models_dict.update({"error_corrected": council})

    # Define testing scenarios
    scenarios = [
        "mlp_mnist_shallowaf_adv",
        "mlp_mnist_shallow_adv",
        "mlp_mnist_deep_adv",
        "mlp_mnist_deepaf_adv",
        "resmlp_mnist_shallowaf_adv",
        "resmlp_mnist_shallow_adv",
        "resmlp_mnist_deep_adv",
        "resmlp_mnist_deepaf_adv",
        # "ae_dim256_rank20",
        # "ae_dim512_rank10",
        # "ae_dim512_rank20",
        # "ae_dim1024_rank10",
        # "ae_dim1024_rank20",
        # "ae_dim2048_rank10",
        # "ae_dim2048_rank20",
    ]
    attack_summary = NestedDefaultDict()

    for attack in attacks:
        attack_name = attack.__class__.__name__

        for scenario in scenarios:
            # Run the attack
            _, clipped_advs_per_epsilon, success = attack(
                all_models_dict[scenario], images, labels, epsilons=epsilons
            )
            assert success.shape == (len(epsilons), len(images))

            for i, adv_tensor in enumerate(clipped_advs_per_epsilon):
                for model_name, eval_model in all_models_dict.items():
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

    # Load MLPs
    model_names = [
        "mlp_mnist_shallowaf_adv",
        "mlp_mnist_shallow_adv",
        "mlp_mnist_deep_adv",
        "mlp_mnist_deepaf_adv",
        "resmlp_mnist_shallowaf_adv",
        "resmlp_mnist_shallow_adv",
        "resmlp_mnist_deep_adv",
        "resmlp_mnist_deepaf_adv",
    ]
    models = [load_model(mlp_path, name)[0].to(device) for name in model_names]

    # Load autoencoders
    ae_names = [
        # f"dim_{256}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model",
        # f"dim_{512}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        # f"dim_{512}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model",
        # f"dim_{1024}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        # f"dim_{1024}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model_adv",
        # f"dim_{2048}_k_{1}_loc_{0}_lowrank_{10}_autoencoder_mnist_model",
        # f"dim_{2048}_k_{1}_loc_{0}_lowrank_{20}_autoencoder_mnist_model_adv",
    ]
    autoencoders = [load_autoencoder(ae_path, name)[0].to(device) for name in ae_names]

    # Load dataset
    _, model_metadata = load_model(mlp_path, model_names[0])
    dataset = load_dataset(model_metadata, data_path)

    # Attack
    results = adv_attack(
        dataset=dataset,
        model_names=model_names,
        ae_names=ae_names,
        models=models,
        autoencoders=autoencoders,
        device=device,
    )

    # Writing to JSON file
    logger.info("Dumping.")
    with open("../adv_output/results.json", "w+") as json_file:
        json.dump(results, json_file, indent=4)

    # Plot results
    logger.info("Plotting.")
    plot_attack_results(results)


if __name__ == "__main__":
    main()
