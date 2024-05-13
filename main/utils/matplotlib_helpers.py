import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import os

column_width = 3.250  # in inches
double_column_width = 6.750  # in inches


def set_default_paras(
    tick_size: int = 7,
    axes_size: int = 7,
    legend_size: int = 7,
    double_width: bool = False,
    width_height: tuple = None,
):
    """Define some sensible matplotlib default parameters that can be imported / used in other places

    Args:
        tick_size (int, optional): Tick label size.
        axes_size (int, optional): Axis label size.
        legend_size (int, optional): Legend label size.
        double_width (bool, optional): Whether to use double column width. Defaults to False.
        width_height (float, optional): Overwrite the width and height of the figure in inches. Defaults to None.
    """

    plt.ioff()

    jobid = os.getenv("SLURM_JOB_ID")
    if jobid is None or jobid == "":
        mpl.rcParams["text.usetex"] = True
    # Use computer modern font (sans-serif)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["mathtext.fontset"] = "cm"

    plt.rc("ytick", labelsize=tick_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("axes", labelsize=axes_size)
    plt.rc("legend", **{"fontsize": legend_size})

    if width_height is not None:
        matplotlib.rcParams["figure.figsize"] = width_height
    else:
        if double_width:
            matplotlib.rcParams["figure.figsize"] = (
                double_column_width,
                1 / 1.618 * double_column_width,
            )
        else:
            matplotlib.rcParams["figure.figsize"] = (
                column_width,
                1 / 1.618 * column_width,
            )


def set_content_to_rasterized(fig):
    for ax in fig.get_axes():
        for child in ax.get_children():
            if isinstance(
                child, (matplotlib.lines.Line2D, matplotlib.collections.Collection)
            ):
                child.set_rasterized(True)
