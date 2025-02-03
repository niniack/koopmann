from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from jaxtyping import Float, Int
from matplotlib.ticker import MaxNLocator
from numpy import ndarray
from torch import Tensor, nn

from koopmann import aesthetics
from koopmann.models.utils import pad_act

# if TYPE_CHECKING:
#     from pykoopman import Koopman


# def plot_prediction(
#     states: Float[Tensor, "batch state iteration"],
#     koopman_scaled_model: Any,
#     sample_idx: Int,
#     plot_num_states=2,
#     figsize=(12, 12),
# ):
#     # number of iterations as the x-axis (time)
#     n_delays = koopman_scaled_model._n_delays
#     n_iters = states.size(dim=-1)
#     t = np.arange(0, n_iters, 1)

#     Xkoop = koopman_scaled_model.simulate(x0=states)[sample_idx].unsqueeze(1)
#     x0_td = states[sample_idx, :, : n_delays + 1]

#     # stack given with predction
#     Xkoop = np.hstack([x0_td, Xkoop])

#     plot_num_states = min(plot_num_states, x0_td.shape[0])
#     fig, axes = plt.subplots(
#         (plot_num_states + 1) // 2,
#         2,
#         sharex=True,
#         tight_layout=False,
#         figsize=figsize,
#         squeeze=False,
#     )
#     axes = axes.flatten()  # Flatten the axes array for easier indexing

#     for state_idx in range(plot_num_states):
#         ax = axes[state_idx]

#         # Plot the original activations
#         ax.plot(t, states[sample_idx, state_idx, :], "-", color="b", label="True States")
#         # Plot the predicted activations
#         ax.plot(t, Xkoop[state_idx, :], "--r", label="Hankel DMD")

#         # Set label, delay line, and ticks
#         ax.axvline(x=t[n_delays], color="gray", linestyle="--")
#         ax.text(
#             0.93,
#             0.9,
#             f"S{state_idx}",
#             transform=ax.transAxes,
#         )
#         ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
#         if state_idx == 0:
#             ax.legend()

#     # Remove any unused subplots
#     for idx in range(plot_num_states, len(axes)):
#         fig.delaxes(axes[idx])


def plot_eigenvalues(
    eigenvalues_dict: dict[tuple, torch.Tensor],
    tile_size: int = 4,
    num_rows: int = -1,
    num_cols: int = -1,
):
    """
    Plot eigenvalues for a dictionary of tensors with tuple keys.

    Args:
        eigenvalues_dict (dict): A dictionary where keys are tuples and values are tensors.
        tile_size (int): Size of each tile in the grid.
    """
    # Calculate grid size
    num_plots = len(eigenvalues_dict)
    if num_rows == -1 or num_cols == -1:
        num_rows = int(np.ceil(np.sqrt(num_plots)))
        num_cols = int(np.ceil(num_plots / num_rows))

    # Create the figure and axes
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(tile_size * num_cols, tile_size * num_rows), squeeze=False
    )

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    # Iterate through the dictionary and plot each set of eigenvalues
    for i, ((key, eigenvalues), ax) in enumerate(zip(eigenvalues_dict.items(), axes)):
        aesthetics.set_spine_color(ax)
        aesthetics.set_equal_aspect(ax)
        ax.set_title(rf"$k={key[0]}$, $dim={key[1]}$", fontsize=12)

        # Plot the unit circle
        unit_circle = plt.Circle(
            (0, 0), 1, color=aesthetics.SeabornColors.blue, fill=False, linestyle="--"
        )
        ax.add_artist(unit_circle)

        # Plot the eigenvalues with reduced alpha for transparency
        sns.scatterplot(
            x=eigenvalues.real.cpu().detach().numpy(),
            y=eigenvalues.imag.cpu().detach().numpy(),
            color=aesthetics.SeabornColors.orange,
            edgecolor=None,
            s=30,
            marker="o",
            alpha=0.7,  # Adjust alpha to reduce blotchiness
            ax=ax,
        )

        # Set the axis limits
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])

        # Set the ticks on both axes
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

    # Hide unused axes
    for ax in axes[num_plots:]:
        ax.axis("off")

    return fig, axes


def plot_decision_boundary(
    model: nn.Module,  # PyTorch model
    final_state_dict: dict,  # Final state dict of the model
    X: torch.Tensor,  # Input data
    y: torch.Tensor,  # Label vector
    labels: list[int] = [0, 1],  # Labels
    ax=None,  # Optional Axes object
) -> None:
    # Use provided Axes or create a new one
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None  # No new figure created

    # Aesthetics
    aesthetics.set_equal_aspect(ax)

    # Generate a color palette with as many colors as there are labels
    colors = sns.color_palette("tab10", len(labels))

    # Initialization
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    x_in = torch.tensor(x_in, dtype=torch.float32).to(next(model.parameters()).device)

    # Plot data points
    for label, color in zip(labels, colors):
        sns.scatterplot(
            x=X[y == label, 0].cpu().numpy(),
            y=X[y == label, 1].cpu().numpy(),
            ax=ax,
            color=color,
            marker="o",
            s=50,
            label=f"Class {label}",
        )

    # Load final model state and set to eval mode
    model.load_state_dict(final_state_dict)
    model.eval()

    # Get predictions on grid points
    with torch.no_grad():
        out = model.forward(x_in)
        y_pred = torch.argmax(out, dim=1).cpu().numpy().reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(
        xx,
        yy,
        y_pred,
        levels=np.arange(len(labels) + 1) - 0.5,  # Adjust levels for proper class separation
        colors=colors,
        alpha=0.5,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Turn off ticks
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    return fig, ax


def plot_koopman_decision_boundary(
    model: nn.Module,  # PyTorch model
    final_state_dict: dict,  # Final state dict of the model
    autoencoder: nn.Module,  # Koopman autoencoder
    X: torch.Tensor,  # Input data
    y: torch.Tensor,  # Label vector
    labels: list[int] = [0, 1],  # Labels
    ax=None,
) -> None:
    # Use provided Axes or create a new one
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None  # No new figure created

    aesthetics.set_equal_aspect(ax)

    # Generate a color palette with as many colors as there are labels
    colors = sns.color_palette("tab10", len(labels))

    # Initialization
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    x_in = torch.tensor(x_in, dtype=torch.float32).to(next(model.parameters()).device)

    # Plot data points
    for label, color in zip(labels, colors):
        sns.scatterplot(
            x=X[y == label, 0].cpu().numpy(),
            y=X[y == label, 1].cpu().numpy(),
            ax=ax,
            color=color,
            marker="o",
            s=50,
            label=f"Class {label}",
        )

    # Load final model state and set to eval mode
    model.load_state_dict(final_state_dict)
    model.eval()

    # Get predictions on grid points
    with torch.no_grad():
        _ = model.forward(x_in)
        acts = model.get_fwd_activations()

    k = autoencoder.steps
    all_pred = autoencoder(
        x=pad_act(acts[0], target_size=autoencoder.encoder[0].in_features), k=k
    ).predictions
    y_pred = all_pred[k, :, : acts[4].size(-1)]
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy().reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(
        xx,
        yy,
        y_pred,
        levels=np.arange(len(labels) + 1) - 0.5,  # Adjust levels for proper class separation
        colors=colors,
        alpha=0.5,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tick_params(
        axis="both",  # Apply changes to both x-axis and y-axis
        which="both",  # Affect both major and minor ticks
        bottom=False,  # Turn off ticks along the bottom edge
        top=False,  # Turn off ticks along the top edge
        left=False,  # Turn off ticks along the left edge (y-axis)
        right=False,  # Turn off ticks along the right edge (y-axis)
        labelbottom=False,  # Turn off labels on the bottom edge (x-axis)
        labelleft=False,  # Turn off labels on the left edge (y-axis)
    )

    return fig, ax
