import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn

from koopmann import aesthetics


def plot_eigenvalues(
    eigenvalues_dict: dict[tuple, torch.Tensor],
    tile_size: int = 4,
    num_rows: int = -1,
    num_cols: int = -1,
    axis: list[int] = [-3, 3],
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
        ax.set_xlim(axis)
        ax.set_ylim(axis)

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
