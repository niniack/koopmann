# aesthetics.py
# Maybe not the best approach, but I like to just load this file up in a notebook
# to get roughly consistent aesthetics when building figures.

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import matrepr
import pyfonts
import seaborn as sns
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeabornColors:
    blue: tuple
    dark_orange: tuple
    orange: tuple
    green: tuple
    black: tuple
    white: tuple


def _make_palette() -> SeabornColors:
    """Create a small, named palette from seaborn's tab20c."""
    palette = sns.color_palette("tab20c")
    return SeabornColors(
        blue=palette[1],
        dark_orange=palette[5],
        orange=palette[6],
        green=palette[9],
        black=palette[16],
        white=palette[19],
    )


palette = sns.color_palette("tab20c")
COLORS = _make_palette()
# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------


def set_spine_color(ax=None, color="lightgray"):
    """
    Set all spines to the same color.
    """
    if ax is None:
        ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color(color)


def set_equal_aspect(ax=None):
    """
    Set equal aspect ratio on an axes.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")


def kill_ticks(ax=None):
    """
    Remove tick labels and tick marks on all visible axes.
    Works for 2D and (if present) 3D axes.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    if hasattr(ax, "set_zticklabels"):
        ax.set_zticklabels([])
    if hasattr(ax, "set_zticks"):
        ax.set_zticks([])


def kill_axes(ax=None, pane=True, edge=True, line=True):
    """
    Hide panes / edges / axis lines, mainly for 3D plots.

    pane=True  -> make panes transparent
    edge=True  -> remove pane edges
    line=True  -> hide axis lines
    """
    if ax is None:
        ax = plt.gca()

    # 2D axes do not have zaxis; handle both cases gracefully.
    axes = [ax.xaxis, ax.yaxis]
    if hasattr(ax, "zaxis"):
        axes.append(ax.zaxis)

    for axis in axes:
        # Some axis types (e.g. 3D) expose a 'pane' attribute.
        if hasattr(axis, "pane"):
            axis.pane.fill = not pane  # pane=True -> transparent
            axis.pane.set_edgecolor("none" if edge else "lightgray")
        if hasattr(axis, "line"):
            axis.line.set_color("none" if line else "black")


# Set the aesthetic parameters in one step using Seaborn
sns.set_theme(style="white", context="paper")
sns.set_style("white")
sns.axes_style("darkgrid")

# load font
ibmplexsans = pyfonts.load_font(
    font_url="https://github.com/google/fonts/blob/057514444ab92c5819ae66fc91d42ad176a37728/ofl/ibmplexsans/IBMPlexSans-Medium.ttf?raw=true"
)
font_manager.fontManager.addfont(ibmplexsans.get_file())

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------


def _load_ibm_plex_sans():
    """
    Load IBM Plex Sans via pyfonts, if possible.
    If this fails (e.g. no internet), we silently fall back to default fonts.
    """
    try:
        ibmplexsans = pyfonts.load_font(
            font_url=(
                "https://github.com/google/fonts/blob/"
                "057514444ab92c5819ae66fc91d42ad176a37728/"
                "ofl/ibmplexsans/IBMPlexSans-Medium.ttf?raw=true"
            )
        )
        font_manager.fontManager.addfont(ibmplexsans.get_file())
        return "IBM Plex Sans"
    except Exception:
        # Don't crash if the font can't be fetched; just use default sans-serif.
        return None


def use():
    """
    Apply a consistent plotting style (seaborn + matplotlib rcParams + matrepr).
    Call once per session, or just import this module if you keep the call at the bottom.
    """

    # Seaborn theme
    sns.set_theme(style="whitegrid", context="poster")

    # Font settings
    font_family = _load_ibm_plex_sans() or "sans-serif"

    rc_updates = {
        # Font
        "font.family": font_family,
        "text.usetex": False,  # Broken :(
        # Tick / legend sizes
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        # Figure size
        "figure.figsize": [10, 10],
        # Grid
        # "grid.color": "#dcdcdc",
        # Savefig
        "savefig.dpi": 300,
    }

    mpl.rcParams.update(rc_updates)

    # Matrix printing (matrepr)
    matrepr.params.max_rows = 30
    matrepr.params.max_cols = 30
    matrepr.params.floatfmt = ".4f"
    matrepr.params.num_after_dots = 5


use()
