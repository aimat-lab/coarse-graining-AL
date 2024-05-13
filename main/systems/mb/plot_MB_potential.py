""" Plot the Muller-Brown potential with the CG axis (overview figure).
"""

import matplotlib.pyplot as plt
import numpy as np
from main.systems.mb.potential import muller_potential
from main.utils.matplotlib_helpers import set_default_paras
import matplotlib.colors as mcolors
import main.utils.matplotlib_helpers

if __name__ == "__main__":
    set_default_paras()

    minx = -1.7
    maxx = 1.13
    miny = -0.4
    maxy = 2.05

    fig = plt.figure(
        figsize=(
            main.utils.matplotlib_helpers.column_width,
            1 / 1.618 * main.utils.matplotlib_helpers.column_width,
        )
    )

    ax = plt.gca()

    grid_width = max(maxx - minx + 0.6, maxy - miny + 0.6) / 630.0
    xx, yy = np.mgrid[
        minx - 0.3 : maxx + 0.3 : grid_width, miny - 0.3 : maxy + 0.3 : grid_width
    ]
    V = 0.1 * muller_potential(xx, yy)  # beta = 0.1
    min_V = np.min(V)
    V = V - min_V

    max_to_plot = 15.0

    # plt.contourf(
    #    xx, yy, V.clip(max=max_to_plot), 15, cmap="nipy_spectral", linewidths=1.0
    # )

    # Copy the original colormap
    original_cmap = plt.cm.nipy_spectral
    colors = original_cmap(np.linspace(0, 1, 256))
    colors = np.vstack(
        (colors, [1, 1, 1, 1])
    )  # Append a white color for over-range values
    new_cmap = mcolors.LinearSegmentedColormap.from_list("custom_nipy_spectral", colors)
    # Set the 'over' property of the colormap to white
    new_cmap.set_over("white")

    plt.imshow(
        V.T,  # your 2D potential energy array
        extent=[
            xx.min(),
            xx.max(),
            yy.min(),
            yy.max(),
        ],  # set the limits of the x and y axes
        origin="lower",
        cmap=new_cmap,
        interpolation="nearest",
        vmax=max_to_plot,  # max value for the colorbar
        rasterized=True,
    )

    cbar = plt.colorbar()
    cbar.set_label(r"Potential energy / $kT$")

    # draw the additional axis

    ref_point = np.array([-2.25, 1.9])
    direction = np.array([1, -1])
    start_point = ref_point - 1 * direction
    end_point = ref_point + 3 * direction

    ax.plot(
        np.array([start_point[0], end_point[0]]),
        np.array([start_point[1], end_point[1]]),
        color="blue",
    )

    def plot_tick(s):
        # s = x - y

        # The line above:
        # x = start_point[0] + t
        # y = start_point[1] - t

        # => s = start_point[0] - start_point[1] + 2*t

        # t = (s - start_point[0] + start_point[1]) / 2

        point = np.array(
            [
                start_point[0] + (s - start_point[0] + start_point[1]) / 2,
                start_point[1] - (s - start_point[0] + start_point[1]) / 2,
            ]
        )

        ax.plot(
            [point[0] - 0.03, point[0] + 0.03],
            [point[1] - 0.03, point[1] + 0.03],
            color="blue",
        )

        ax.annotate(
            r"\textbf{" + "{:.1f}".format(s) + "}",
            xy=point,
            xytext=(-18 + (3 if s >= 0 else 0), -10 + (4 if s >= 0 else 0)),
            textcoords="offset points",
            color="blue",
            fontsize=7,
            weight="bold",
        )

    plot_tick(0.0)
    plot_tick(-1.0)
    plot_tick(-2.0)

    ax.annotate(
        "s",
        xy=[-0.60, -0.5],
        xytext=(-28, 12),
        textcoords="offset points",
        color="blue",
        fontsize=7,
    )
    # ax.axis("equal")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    plt.tight_layout()
    plt.savefig("potential.pdf", dpi=600, bbox_inches="tight")
