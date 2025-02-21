import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy

from koopmann.data import DatasetConfig, MNISTDataset
from koopmann.models import MLP, Autoencoder
from koopmann.utils import compute_model_accuracy

user = "nsa325"


def fgsm_attack(model, loss_fn, data, epsilon, target):
    """
    FGSM attack implementation
    """
    # Make data require gradients
    data.requires_grad = True

    # Forward pass
    outputs = model(data)

    # Calculate loss
    loss = loss_fn(outputs, target)

    # Get gradients
    loss.backward()

    # Collect gradients
    data_grad = data.grad.data

    # Create perturbation
    sign_data_grad = data_grad.sign()

    # Create adversarial example
    perturbed_data = data + epsilon * sign_data_grad

    # Clamp to ensure valid pixel range [0,1]
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data


def visualize_results(results, save_path):
    """Visualize the results"""
    # Create accuracy comparison plot
    plt.figure(figsize=(10, 6))
    accuracies = [
        results["original_accuracy"],
        results["adversarial_accuracy"],
        results["autoencoder_accuracy"],
    ]
    labels = ["Original", "Adversarial", "Autoencoder Protected"]

    sns.barplot(x=labels, y=accuracies)
    plt.title(f"Accuracy Comparison (ε={results['epsilon']})")
    plt.ylabel("Accuracy (%)")
    plt.savefig(save_path / f"accuracy_comparison_epsilon_{results['epsilon']}.png")
    plt.close()


def test_autoencoder_robustness(args):
    data_root = Path(f"/scratch/{user}/datasets")
    model_path = Path(f"/scratch/{user}/koopmann_model_saves")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_samples = 60_000
    epsilon = args.epsilon

    dataset_config = DatasetConfig(
        dataset_name="MNISTDataset",
        num_samples=n_samples,
        split="test",
        torch_transform=None,
        seed=42,
    )

    try:
        test_dataset = MNISTDataset(
            config=dataset_config,
            root=str(data_root),
            seed=42,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1_024,
            shuffle=False,
        )
        print(f"Successfully loaded dataset with {len(test_loader)} batches")

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

    # Load models
    try:
        # modify the file path
        autoencoder_path = (
            model_path / "scaling/k_1_dim_1024_loc_0_autoencoder_mnist_model.safetensors"
        )
        mlp_path = model_path / "mnist_probed.safetensors"

        # check file path exist?
        if not autoencoder_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_path}")

        if not mlp_path.exists():
            raise FileNotFoundError(f"MLP model not found: {mlp_path}")

        print(f"Loading autoencoder from: {autoencoder_path}")
        autoencoder, _ = Autoencoder.load_model(autoencoder_path)
        autoencoder = autoencoder.to(device)
        autoencoder.eval()

        print(f"Loading MLP from: {mlp_path}")
        mlp, _ = MLP.load_model(mlp_path)

        # NOTE: Remove nonlinearities from the probe!
        mlp.modules[-2].remove_nonlinearity()
        mlp.modules[-3].remove_nonlinearity()

        mlp.to(device).eval()

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    mlp_original_metric = MulticlassAccuracy()
    mlp_adversarial_metric = MulticlassAccuracy()
    koopman_aug_metric = MulticlassAccuracy()

    # Run tests
    for i, (data, target) in enumerate(test_loader):
        if i >= 1:
            break

        data, target = data.to(device), target.to(device)
        data = data.flatten(start_dim=1)

        # Generate adversarial example
        perturbed_data = fgsm_attack(mlp, nn.CrossEntropyLoss(), data, epsilon, target)

        with torch.no_grad():
            # Test MLP with original data
            original_output = mlp(data)
            mlp_original_metric.update(original_output, target.squeeze())

            # Test MLP with adversarial data
            adv_output = mlp(perturbed_data)
            mlp_adversarial_metric.update(adv_output, target.squeeze())

            # Test with autoencoder's latent representation
            predict_act = autoencoder(perturbed_data, k=1).predictions[-1]
            koopman_aug_output = mlp.modules[-2:](predict_act)
            koopman_aug_metric.update(koopman_aug_output, target.squeeze())

    mlp_original_acc = mlp_original_metric.compute()
    mlp_adv_acc = mlp_adversarial_metric.compute()
    koopman_acc = koopman_aug_metric.compute()

    print(f"Original accuracy for a single batch: {mlp_original_acc}")
    print(f"Adversarial attack accuracy for a single batch: {mlp_adv_acc}")
    print(f"Koopman accuracy for a single batch: {koopman_acc}")

    # Save results
    results = {
        "original_accuracy": mlp_original_acc.item(),
        "adversarial_accuracy": mlp_adv_acc.item(),
        "autoencoder_accuracy": -1,
        "epsilon": epsilon,
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.1, help="Perturbation magnitude")

    args = parser.parse_args()

    try:
        test_autoencoder_robustness(args)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
