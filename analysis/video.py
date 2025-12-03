import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from safetensors import safe_open
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torcheval.metrics import MulticlassAccuracy
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from analysis.utils import load_autoencoder, load_model
from koopmann import aesthetics
from koopmann.data import DatasetConfig, get_dataset_class
from koopmann.models import ConvResNet
from koopmann.shape_metrics import prepare_acts, undo_preprocessing_acts


def get_dataset_config(dataset_name):
    configs = {
        "lotusroot": {"dim": 20, "scale_idx": 1, "k_steps": 100, "flavor": "exponential"},
        "torus": {"dim": 50, "scale_idx": 1, "k_steps": 100, "flavor": "exponential"},
        "yinyang": {"dim": 20, "scale_idx": 1, "k_steps": 100, "flavor": "exponential"},
        "mnist": {"dim": 800, "scale_idx": 1, "k_steps": 10, "flavor": "exponential"},
        "cifar10": {"dim": 1000, "scale_idx": 1, "k_steps": 5000, "flavor": "exponential"},
    }
    return configs[dataset_name]


def process_pca_and_align(data, reference):
    pca_result = PCA(n_components=3).fit_transform(data)
    _, aligned, _ = procrustes(reference, pca_result)
    return aligned


def create_combined_plot(data_a, data_b, labels, axis_range=[-0.02, 0.02]):
    """Create side-by-side 3D scatter plots using matplotlib."""
    # Set IBM Plex Sans font
    plt.rcParams["font.family"] = aesthetics.ibmplexsans.get_name()

    fig = plt.figure(figsize=(12, 6))

    # Custom color palette
    # palette = ["#fc9918", "#a779aa", "#4c3191"]
    palette = list(mcolors.TABLEAU_COLORS.values())
    colors = np.array(labels).flatten()
    color_map = [palette[int(c) % len(palette)] for c in colors]

    # Left subplot - View A
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(data_a[:, 0], data_a[:, 1], data_a[:, 2], c=color_map, s=40)  # Increased size
    ax1.set_xlim(axis_range)
    ax1.set_ylim(axis_range)
    ax1.set_zlim(axis_range)
    ax1.set_title("View A", y=0.90, fontsize=18)  # Lower title position
    ax1.view_init(elev=20, azim=45)  # Default angle
    ax1.set_axis_off()

    # Right subplot - View B
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(data_b[:, 0], data_b[:, 1], data_b[:, 2], c=color_map, s=40)  # Increased size
    ax2.set_xlim(axis_range)
    ax2.set_ylim(axis_range)
    ax2.set_zlim(axis_range)
    ax2.set_title("View B", y=0.90, fontsize=18)  # Lower title position
    ax2.view_init(elev=30, azim=120)  # Different angle
    ax2.set_axis_off()

    plt.tight_layout()
    return fig


def generate_frame(layer_index, ae_predictions, ae_pred_obs, ref_act, labels):
    """Generate a single frame and return as numpy array."""
    data_a = process_pca_and_align(ae_predictions[layer_index].cpu(), ref_act)
    data_b = process_pca_and_align(ae_pred_obs[layer_index].cpu(), ref_act)

    fig = create_combined_plot(data_a, data_b, labels)

    # Main title
    fig.suptitle("Decoded States from a Koopman Autoencoder", fontsize=20, y=0.95)

    # Progress bar
    total_frames = ae_predictions.shape[0]
    progress = layer_index / (total_frames - 1)  # 0 to 1
    max_bar_width = 0.3  # Maximum width of the progress bar
    current_bar_width = progress * max_bar_width

    # Create progress bar that grows from center
    bar_center_x = 0.5
    bar_left = bar_center_x - current_bar_width / 2
    bar_y = 0.12
    bar_height = 0.008

    # Add progress bar background (light gray)
    bg_left = bar_center_x - max_bar_width / 2
    bg_rect = patches.Rectangle(
        (bg_left, bar_y),
        max_bar_width,
        bar_height,
        transform=fig.transFigure,
        facecolor="#e0e0e0",
        alpha=0.3,
    )
    fig.patches.append(bg_rect)

    # Add progress bar (lavender)
    if current_bar_width > 0:
        progress_rect = patches.Rectangle(
            (bar_left, bar_y),
            current_bar_width,
            bar_height,
            transform=fig.transFigure,
            facecolor="#b19cd9",
            alpha=0.8,
        )
        fig.patches.append(progress_rect)

    # Iteration counter at bottom (higher y position so it's visible)
    fig.text(0.5, 0.08, f"Iteration {layer_index}", fontsize=16, ha="center")

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    buf = np.asarray(buf)
    width, height = fig.canvas.get_width_height()
    buf = buf.reshape(height, width, 4)[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    return buf


def create_video(frames, output_path, fps=5):
    """Create GIF from frame arrays - more compatible."""
    gif_path = output_path.replace(".mp4", ".gif")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {gif_path}")


def main():
    # Configuration
    dataset_name = "mnist"
    file_dir = "/home/dunstan/git/koopmann/models"
    data_root = "./datasets/"
    device = "cpu"
    subset_size = None
    video_output = "./pca_animation.mp4"

    config = get_dataset_config(dataset_name)
    dim, scale_idx, k_steps, flavor = (
        config["dim"],
        config["scale_idx"],
        config["k_steps"],
        config["flavor"],
    )

    model_name = f"resmlp_{dataset_name}"
    ae_name = f"{dataset_name}/dim_{dim}_k_{k_steps}_loc_{scale_idx}_{flavor}_autoencoder_{dataset_name}_model_seed_{21}"

    # Load models
    model, model_metadata = load_model(file_dir, model_name)
    model.eval().hook_model().to(device)

    autoencoder, ae_metadata = load_autoencoder(file_dir, ae_name)
    autoencoder.eval().to(device)
    preprocess = ae_metadata["preprocess"]

    # Setup dataset
    dataset_config = DatasetConfig(
        dataset_name=model_metadata["dataset"], num_samples=3_000, split="test", seed=42
    )
    dataset = get_dataset_class(name=dataset_config.dataset_name)(
        config=dataset_config, root=data_root
    )

    batch_size = min(subset_size, 3_000) if subset_size else 3_000
    dataloader = DataLoader(
        Subset(dataset, list(range(subset_size))) if subset_size else dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Load preprocessing
    preproc_dict = {}
    with safe_open(
        f"{file_dir}/{ae_name}_preprocessing.safetensors", framework="pt", device="cpu"
    ) as f:
        for k in f.keys():
            preproc_dict[k] = f.get_tensor(k)

    # Process activations
    with torch.no_grad():
        orig_act_dict, proc_act_dict, _ = prepare_acts(
            dataloader,
            model,
            device,
            ae_metadata["in_features"],
            preproc_dict["wh_alpha_0"],
            preprocess,
            preproc_dict,
            only_first_last=False,
        )
        if not preprocess:
            proc_act_dict = orig_act_dict

    ref_act = PCA(n_components=3).fit_transform(proc_act_dict[0].cpu())

    # Generate predictions
    init_idx, final_idx = list(orig_act_dict.keys())[0], list(orig_act_dict.keys())[-1]

    with torch.no_grad():
        ae_result = autoencoder(proc_act_dict[init_idx], intermediate=True, k=k_steps)
        ae_pred_obs = torch.stack(
            [
                autoencoder.koopman_forward(autoencoder.encode(pred), 1)
                for pred in ae_result.predictions
            ]
        )

        pred = ae_result.predictions[-1]
        if preprocess:
            pred = undo_preprocessing_acts(pred, preproc_dict, final_idx, device)
        if isinstance(model, ConvResNet):
            pred = pred.reshape(-1, 512, 4, 4)

    # Compute accuracy
    koopman_pred = model.components[-1:](pred)
    accuracy = MulticlassAccuracy(num_classes=dataset.out_features)
    labels_tensor = torch.tensor(dataset.labels[:subset_size], dtype=torch.long).squeeze()
    accuracy.update(koopman_pred, labels_tensor)
    print(f"Koopman accuracy: {accuracy.compute()}")

    # Generate video frames in memory
    labels = dataset.labels if not subset_size else dataset.labels[:subset_size]
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    elif hasattr(labels, "numpy"):
        labels = labels.numpy()

    num_frames = ae_result.predictions.shape[0]

    print(f"Generating {num_frames} frames...")
    frames = []
    for layer_idx in range(num_frames):
        frame = generate_frame(layer_idx, ae_result.predictions, ae_pred_obs, ref_act, labels)
        frames.append(frame)
        if layer_idx % 10 == 0:
            print(f"Generated frame {layer_idx}/{num_frames}")

        # Create video directly from memory
    print("Adding pause frames...")
    # Add 15 duplicate frames at the end for a pause (1.5 seconds at 10fps)
    pause_frames = 5
    for _ in range(pause_frames):
        frames.append(frames[-1])  # Duplicate the last frame

    # Create video directly from memory
    create_video(frames, video_output, fps=5)


if __name__ == "__main__":
    main()
