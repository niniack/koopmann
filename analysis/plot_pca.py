from collections import OrderedDict
from copy import deepcopy
from functools import reduce

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import torch
import typer
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import Subset

from koopmann import aesthetics
from koopmann.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from koopmann.models import MLP, Autoencoder, ExponentialKoopmanAutencoder
from koopmann.models.utils import get_device, pad_act, parse_safetensors_metadata


def get_dataloader(model_file_path: str, batch_size: int = 5_000):
    """Prepare dataloader."""
    # Dataset config
    metadata = parse_safetensors_metadata(file_path=model_file_path)
    dataset_config = DatasetConfig(
        dataset_name=metadata["dataset"],
        num_samples=batch_size,
        split="test",
        seed=21,
    )
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    train_dataset = DatasetClass(config=dataset_config)

    # Reduce to certain class
    # idx = torch.where(train_dataset.labels == target_class)[0]
    # subset = Subset(train_dataset, idx)
    # loader = torch.utils.data.DataLoader(train_dataset, batch_size=10_000, shuffle=True)

    dataloader = create_data_loader(train_dataset, batch_size=batch_size)
    return dataloader


def fit_pca(data):
    """PCA for alignment."""
    pca = PCA(n_components=3)
    ref = pca.fit_transform(data)
    return ref


def process_pca_and_align(data, reference):
    """Applies PCA, aligns using Procrustes, and returns aligned data."""
    _, aligned_result, _ = procrustes(reference, fit_pca(data))
    return aligned_result


def get_activations(model, autoencoder, batch, k_steps, device):
    ################## FORWARD ###############
    input, _ = batch
    with torch.no_grad():
        _ = model(input.to(device))
    act_dict = model.get_fwd_activations()

    ################## STATE SPACE PREDICTIONS ###############
    new_keys = list(range(0, k_steps + 1))
    decoded_act = autoencoder(x=act_dict[0], k=k_steps).predictions.cpu().detach().numpy()
    ref_decoded = fit_pca(decoded_act[0])
    decoded_act = [process_pca_and_align(decoded, ref_decoded) for decoded in decoded_act]
    decoded_act_dict = OrderedDict(zip(new_keys, decoded_act))

    # ################## OBSERVABLE SPACE PREDICTIONS ###############
    embedded_act = [autoencoder.encoder(act_dict[0])] * (k_steps + 1)
    with torch.no_grad():
        embedded_act = [
            act if i == 0 else reduce(lambda x, _: autoencoder.koopman_matrix(x), range(i), act)
            for i, act in enumerate(embedded_act)
        ]
    ref_embedded = fit_pca(embedded_act[0].cpu().detach())
    embedded_act = [
        process_pca_and_align(embedded.cpu().detach().numpy(), ref_embedded)
        for embedded in embedded_act
    ]
    embedded_act_dict = OrderedDict(zip(new_keys, embedded_act))

    return decoded_act_dict, embedded_act_dict


def animate(state_dict, observable_dict, labels, plot_metadata):
    # Unwrap metadata
    k_steps = plot_metadata["k_steps"]
    ae_dim = plot_metadata["latent_dim"]
    dataset_name = plot_metadata["dataset_name"]

    # Generate a Seaborn palette for the number of unique labels
    n_labels = labels.unique().numel()
    colors_palette = torch.tensor(sns.color_palette("tab10", n_labels), dtype=torch.float32)

    # Map labels to colors
    colors = colors_palette[labels]
    marker_size = 3

    # Define axis limits and style
    axis_limits = (-0.03, 0.03)

    # Create figure and 3D subplots
    fig = plt.figure(figsize=(12, 7))  # Wider and shorter figure
    ax1 = fig.add_subplot(121, projection="3d")  # Left subplot
    ax2 = fig.add_subplot(122, projection="3d")  # Right subplot

    # Colors
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#ffffff"
    plt.rcParams["axes.edgecolor"] = "#cccccc"

    # Adjust layout
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.15, wspace=0.3)

    # Customize both axes
    for ax in [ax1, ax2]:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_zlim(axis_limits)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.invert_zaxis()  # Flip z-axis

        # Add axis titles
        ax.set_xlabel("PC 1", fontsize=12, font=aesthetics.ibmplexsans, color="#444444", labelpad=2)
        ax.set_ylabel("PC 2", fontsize=12, font=aesthetics.ibmplexsans, color="#444444", labelpad=2)
        ax.set_zlabel("PC 3", fontsize=12, font=aesthetics.ibmplexsans, color="#444444", labelpad=2)

    # Title and subtitle for the entire figure

    fig.suptitle(
        t="Koopman Scaling",
        fontsize=18,
        font=aesthetics.ibmplexsans,
        color="#444444",
        # y=0.97,
    )

    fig.text(
        x=0.5,
        y=0.92,
        s=rf"{dataset_name}, $k={k_steps}$, $dim={ae_dim}$",
        fontsize=14,
        font=aesthetics.ibmplexsans,
        color="#444444",
        ha="center",
    )
    ax1.set_title("State Space", fontsize=12, color="#444444")
    ax2.set_title("Observable Space", fontsize=12, color="#444444")

    # Initialize scatter plot placeholders with dummy data
    x, y, z = state_dict[0][:, 0], state_dict[0][:, 1], state_dict[0][:, 2]
    scatter1 = ax1.scatter(x, y, z, c=colors, s=marker_size)

    x, y, z = observable_dict[0][:, 0], observable_dict[0][:, 1], observable_dict[0][:, 2]
    scatter2 = ax2.scatter(x, y, z, c=colors, s=marker_size)

    # Add a slider for showing progress
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.02], facecolor="blue")
    slider = Slider(ax_slider, r"$k$-steps", 0, k_steps - 1, valinit=0, valstep=1)

    # Update function for animation and slider
    def update(frame):
        # Update scatter plot data for both subplots
        scatter1._offsets3d = (
            state_dict[frame][:, 0],
            state_dict[frame][:, 1],
            state_dict[frame][:, 2],
        )

        scatter2._offsets3d = (
            observable_dict[frame][:, 0],
            observable_dict[frame][:, 1],
            observable_dict[frame][:, 2],
        )

        # Update the rotation of the camera
        ax1.view_init(elev=30, azim=frame * (1 / k_steps))  # Rotate camera for ax1
        ax2.view_init(elev=30, azim=frame * (1 / k_steps))  # Opposite rotation for ax2

        # Sync slider value
        slider.set_val(frame)

        return scatter1, scatter2

    def slider_update(val):
        frame = int(slider.val)
        scatter1._offsets3d = (
            state_dict[frame][:, 0],
            state_dict[frame][:, 1],
            state_dict[frame][:, 2],
        )
        scatter2._offsets3d = (
            observable_dict[frame][:, 0],
            observable_dict[frame][:, 1],
            observable_dict[frame][:, 2],
        )

        ax1.view_init(elev=30, azim=frame * (1 / k_steps))  # Rotate camera for ax1
        ax2.view_init(elev=30, azim=frame * (1 / k_steps))  # Opposite rotation for ax2
        fig.canvas.draw_idle()

    slider.on_changed(slider_update)

    # Animation
    ani = FuncAnimation(fig, update, frames=k_steps, blit=False)

    return ani


def main(model_name: str):
    ################## FILENAME ###############
    scale_idx = 1
    k = 50
    dim = 20
    fps = 10
    model_file_path = f"/home/nsa325/work/koopmann/model_saves/{model_name}.safetensors"
    ae_file_path = f"/scratch/nsa325/koopmann/k_{k}_dim_{dim}_loc_{scale_idx}_autoencoder_{model_name}.safetensors"
    device = get_device()
    print(f"Using device {device}")

    #################### DATA #################
    dataloader = get_dataloader(model_file_path)
    batch = next(iter(dataloader))
    input, labels = batch

    ################### MODELS ################
    model, model_metadata = MLP.load_model(model_file_path)
    model.to(device).eval().hook_model()
    # autoencoder, ae_metadata = Autoencoder.load_model(ae_file_path)
    autoencoder, ae_metadata = ExponentialKoopmanAutencoder.load_model(ae_file_path)
    autoencoder.to(device).eval()
    k_steps = int(ae_metadata["num_scaled"])

    ################# ACTIVATIONS ##############
    decoded_act_dict, embedded_act_dict = get_activations(
        model, autoencoder, batch, k_steps, device
    )

    ################## ANIMATION ###############
    plot_metadata = {
        "k_steps": k_steps,
        "latent_dim": ae_metadata["latent_dimension"],
        "dataset_name": model_metadata["dataset"],
    }
    ani = animate(decoded_act_dict, embedded_act_dict, labels, plot_metadata)
    # Save as MP4
    ani.save(
        f"../animations/pca_k{k}_dim{dim}_fps{fps}.mov",
        writer="ffmpeg",
        fps=fps,
        extra_args=["-vcodec", "prores_ks", "-profile:v", "3"],
    )
    # Save as MP4
    ani.save(
        f"../animations/pca_k{k}_dim{dim}_fps{fps}.mp4",
        writer="ffmpeg",
        fps=fps,
        extra_args=["-vcodec", "mpeg4", "-q:v", "3", "-b:v", "3M"],
    )

    # Save as GIF
    ani.save(
        f"../animations/pca_k{k}_dim{dim}_fps{fps}.gif",
        writer="pillow",
        fps=fps,
    )


if __name__ == "__main__":
    typer.run(main)
