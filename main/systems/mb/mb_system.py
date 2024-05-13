from main.systems.system import System
import torch
from main.systems.mb.potential import muller_potential
import numpy as np
import wandb
import matplotlib.pyplot as plt
from typing import Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from main.utils.call_model_batched import call_model_batched
import main.utils.matplotlib_helpers
from main.utils.plot_free_energy import plot_free_energy_histogram
import io
import PIL.Image as Image
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
from main.systems.mb.potential import get_ground_truth_free_energies
import os
from main.models.flow_base import ConditionalFlowBase
import logging
from main.utils.matplotlib_helpers import set_content_to_rasterized
import matplotlib.path as mpath


def plot_1D_free_energy_from_histogram(
    hs: np.ndarray,
    ax=None,
    color: str = None,
    weights: np.ndarray = None,
    label: str = "Histogram",
) -> plt.Figure:
    """Plot a 1D free energy plot of predicted F values and histogram values.

    Args:
        hs (np.ndarray): Array of hs
        ax (matplotlib.axes._subplots.AxesSubplot): Axes object to plot on. Defaults to None.
        color (str): Color to use for the plot. Defaults to None.
        weights (np.ndarray): Weights for the histogram. Defaults to None.
        label (str): Label for the plot. Defaults to None.

    Returns:
        plt.Figure: Figure object.
    """

    if ax is None:
        fig = plt.figure()
    else:
        fig = ax.get_figure()

    def _plot_single(hs, label="Histogram", ax=None):
        hist, edges = np.histogram(hs[:, 0], 100, weights=weights)
        x = 0.5 * (edges[:-1] + edges[1:])

        prob = hist / np.sum(hist)

        free_energy = np.inf * np.ones(shape=prob.shape)

        nonzero = hist.nonzero()
        free_energy[nonzero] = -np.log(prob[nonzero]) * (
            1.0  # We plot the free energy in units of 1/beta
        )

        free_energy[nonzero] -= np.min(free_energy[nonzero])

        # ax.plot(x[nonzero], free_energy[nonzero], label=label, color=color)
        ax.plot(x, free_energy, label=label, color=color)

        ax.set_ylabel(r"Free energy / $kT$")

    if ax is None:
        ax = plt.gca()

    if hs is not None:
        _plot_single(hs, label=label, ax=ax)

    return fig


def _get_ground_truth_expected_values_orthogonal_to_s_values(
    s_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the expected values orthogonal to s for a given set of s values.

    Args:
        s_values (np.ndarray): Array of s values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays (mean_x, mean_y).
    """

    a = 1.0
    b = -1.0

    mean_xs = np.zeros_like(s_values)
    mean_ys = np.zeros_like(s_values)

    for i, s in enumerate(s_values):
        y = np.linspace(-1, 3, 100)
        x = s / a - b / a * y

        probs = np.exp(-wandb.config.beta * muller_potential(x, y))
        probs = probs / probs.sum()
        mean_y = np.sum(probs * y)
        mean_x = s / a - b / a * mean_y

        mean_xs[i] = mean_x
        mean_ys[i] = mean_y

    return mean_xs, mean_ys


def _get_model_expected_values_orthogonal_to_s_values(
    s_values: torch.Tensor, cond_flow: ConditionalFlowBase
) -> torch.Tensor:
    """Get the expected values of the model orthogonal to s for a given set of s values.

    Args:
        s_values (torch.Tensor): Tensor of s values.
        cond_flow (ConditionalFlowBase): Conditional flow.

    Returns:
        torch.Tensor: Tensor of expected values orthogonal to s.
    """

    with torch.no_grad():
        mean_s_orth = cond_flow.forward(
            torch.zeros_like(s_values, device="cuda"),  # just pass z = 0 in reverse
            context=s_values,
        )

    mean_x0 = (s_values[:, 0] + mean_s_orth[:, 0]) / 2.0
    mean_x1 = (mean_s_orth[:, 0] - s_values[:, 0]) / 2.0

    mean_x = torch.cat((mean_x0[:, None], mean_x1[:, None]), dim=1)

    return mean_x


def get_model_stdv_orth_to_s(
    s_values: torch.Tensor, cond_flow: ConditionalFlowBase
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the standard deviation of the model orthogonal to s for a given set of s values.

    Args:
        s_values (torch.Tensor): Tensor of s values.
        cond_flow (ConditionalFlowBase): Conditional flow.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (lower_bound of stdv, upper_bound of stdv).
    """

    with torch.no_grad():
        stdv_s_orth_lower = cond_flow.forward(
            torch.ones_like(s_values, device="cuda"),  # pass z = 1 in reverse
            context=s_values,
        )

    lower_bound_x0 = (s_values[:, 0] + stdv_s_orth_lower[:, 0]) / 2.0
    lower_bound_x1 = (stdv_s_orth_lower[:, 0] - s_values[:, 0]) / 2.0

    lower_bound_x = torch.cat((lower_bound_x0[:, None], lower_bound_x1[:, None]), dim=1)

    with torch.no_grad():
        stdv_s_orth_upper = cond_flow.forward(
            -1.0 * torch.ones_like(s_values, device="cuda"),  # pass z = -1 in reverse
            context=s_values,
        )

    upper_bound_x0 = (s_values[:, 0] + stdv_s_orth_upper[:, 0]) / 2.0
    upper_bound_x1 = (stdv_s_orth_upper[:, 0] - s_values[:, 0]) / 2.0

    upper_bound_x = torch.cat((upper_bound_x0[:, None], upper_bound_x1[:, None]), dim=1)

    return lower_bound_x, upper_bound_x


class MBSystem(System):
    def __init__(
        self,
        beta: float,
    ):
        internal_coordinates_DOF = 2
        CG_indices = [
            0,
        ]
        self.beta = beta

        # We don't need a ground truth trajectory, because for this system we can
        # calculate the free energy using numerical integration.
        super().__init__(internal_coordinates_DOF, CG_indices, torch.zeros((0, 2)))

        self.ground_truth_trajectory_CG = None

        # Overwrite self.ground_truth_CG_histogram with the numerically integrated free energy
        ground_truth_free_energies = get_ground_truth_free_energies(
            self.ground_truth_CG_histogram_bin_centers[0],
            beta=beta,
        )
        ground_truth_free_energies -= np.median(ground_truth_free_energies)
        # Calculate probabilities from free energies
        ground_truth_probabilities = np.exp(-ground_truth_free_energies)

        # Normalize over the bins
        ground_truth_probabilities /= np.sum(ground_truth_probabilities)
        self.ground_truth_CG_probabilities = ground_truth_probabilities

        ground_truth_free_energies = -np.log(ground_truth_probabilities)
        self.ground_truth_free_energies = ground_truth_free_energies

    def cartesian_to_internal(self, x_cart: torch.Tensor, output_log_det=False):
        s = x_cart[:, 0:1] - x_cart[:, 1:2]  # s = x - y
        s_orth = x_cart[:, 0:1] + x_cart[:, 1:2]  # s_orthogonal = x + y

        x_int = torch.cat([s, s_orth], dim=1)

        if not output_log_det:
            return x_int
        else:
            return x_int, torch.log(
                2.0 * torch.ones(x_cart.shape[0], device=x_cart.device)
            )

    def internal_to_cartesian(self, x_int: torch.Tensor, output_log_det=False):
        x = 0.5 * (x_int[:, 0:1] + x_int[:, 1:2])  # x = 0.5 * (s + s_orth)
        y = 0.5 * (x_int[:, 1:2] - x_int[:, 0:1])  # y = 0.5 * (s_orth - s)

        x_cart = torch.cat([x, y], dim=1)

        if not output_log_det:
            return x_cart
        else:
            return x_cart, torch.log(
                0.5 * torch.ones(x_int.shape[0], device=x_int.device)
            )

    def target_log_prob(self, x_int: torch.Tensor):
        x_cart, log_det = self.internal_to_cartesian(x_int, output_log_det=True)
        return (
            -self.beta * muller_potential(x_cart[:, 0:1], x_cart[:, 1:2])[:, 0]
            + log_det
        )

    def get_uniform_CG_ranges(self):
        return torch.Tensor([[-2.5, 1.1]])

    def run_analysis(
        self,
        cond_flow: ConditionalFlowBase,
        epoch: int,
        tag: str,
        s_samples: torch.Tensor = None,
        training_points: np.ndarray = None,
        new_points: np.ndarray = None,
        MC_starting_points: torch.Tensor = None,
        plot_means: bool = False,
        free_energy_nbins: int = 50,
        min_max_s: Tuple[float, float] = (-2.5, 1.1),
        free_energy_ensemble: FreeEnergyEnsemble = None,
        output_dir: str = None,
        plot_args: dict = None,
    ):
        """Evaluate the backmapping for a given set of mapped ground-truth h_current.

        Args:
            cond_flow (torch.nn.Module): Conditional flow.
            epoch (int): Epoch number.
            tag (str): Tag to add to the logged images' name.
            s_samples (torch.Tensor, optional): Array of CG s-values used for backmapping.
                Can be None, in which case the samples are generated using rejection sampling.
                Defaults to None.
            training_points (np.ndarray, optional): Points in the AL dataset so far. Defaults to None.
            new_points (np.ndarray, optional): New s-points that have just been added to the training set.
                Defaults to None.
            MC_starting_points (torch.Tensor, optional): Starting points for the MC sampling. Defaults to None.
            plot_means (bool, optional): Whether to plot the mean values orthogonal to s. Defaults to False.
            free_energy_nbins (int, optional): Number of bins for the potential energy backmapped plot. Defaults to 50.
            min_max_s (Tuple[float, float], optional): Min and max s values to use for the rejection sampling.
                Defaults to (-2.5, 1.1).
            free_energy_ensemble (FreeEnergyEnsemble, optional): Free energy ensemble. Defaults to None.
            output_dir (str, optional): Path to save the figure to. If this is None, the figure is logged to wandb.
                Defaults to None.
            plot_args (dict, optional): Dictionary of arguments for the plotting. Defaults to None.
        """

        # region Call free energy model to get predictions

        if free_energy_ensemble is not None:
            # Generate some test data in the range of s
            s_values_predicted = np.linspace(-2.5, 1.1, 100)
            s_values_predicted_tensor = torch.tensor(
                s_values_predicted, dtype=torch.get_default_dtype(), device="cuda"
            ).unsqueeze(1)

            with torch.no_grad():
                # Get the predictions and the standard deviation
                # (
                #    free_energy_values_predicted,
                #    std_dev_predicted,
                # ) = free_energy_ensemble.mean_and_std_dev(
                #    s_values_predicted_tensor, select_force=False
                # )

                predicted_outputs = free_energy_ensemble.forward(
                    s_values_predicted_tensor,
                    select_force=False,
                    same_batch_all_models=True,
                    MC_samples=10,
                    same_seed_MC_samples=True,
                )
                free_energy_values_predicted = torch.mean(predicted_outputs, dim=0)
                std_dev_predicted = torch.std(predicted_outputs, dim=0)

            # Convert to numpy arrays for plotting
            free_energy_values_predicted = free_energy_values_predicted.cpu().numpy()
            free_energy_values_predicted -= np.min(free_energy_values_predicted)
            std_dev_predicted = std_dev_predicted.cpu().numpy()

            if plot_args.get("save_PMF_to_file", False):
                np.save(
                    os.path.join(output_dir, "s_values.npy"),
                    s_values_predicted,
                )
                np.save(
                    os.path.join(output_dir, "PMF.npy"),
                    free_energy_values_predicted,
                )
                np.save(
                    os.path.join(output_dir, "stdv.npy"),
                    std_dev_predicted,
                )

        # endregion

        # region Get s samples if not supplied

        if s_samples is None:
            # Number of points used for backmapping
            N = 500000
            N_batch = 100000

            # Sample in batches of 100000:

            for i in range(0, N // N_batch):
                if i == 0:
                    s_samples = free_energy_ensemble.rejection_sampling_in_range_1D(
                        min_max_s[0], min_max_s[1], N_batch
                    ).cpu()[:, None]
                else:
                    s_samples = torch.cat(
                        (
                            s_samples,
                            free_energy_ensemble.rejection_sampling_in_range_1D(
                                min_max_s[0], min_max_s[1], N_batch
                            ).cpu()[:, None],
                        ),
                        dim=0,
                    )

        # endregion

        # region Backmap the samples

        # Prepare input for backmapping procedure
        z = cond_flow.q0.sample(s_samples.shape[0]).to("cpu")

        helper_fn = lambda z, samples: cond_flow.forward(z, context=samples)

        with torch.no_grad():
            x_int_fg = call_model_batched(
                helper_fn,
                z,
                cond_tensor=s_samples,
                pass_cond_tensor_as_tuple=False,
                device="cuda",
                batch_size=wandb.config.batch_size_probability,
                do_detach=False,
            )

        if x_int_fg is None:
            logging.info(f"Backmapping failed for epoch {epoch}.")
            return

        x_int_backmapped = torch.empty((x_int_fg.shape[0], 2))
        x_int_backmapped[:, self.FG_mask] = x_int_fg
        x_int_backmapped[:, self.CG_mask] = s_samples

        x_int_backmapped = x_int_backmapped.detach().to("cpu")
        x_backmapped = self.internal_to_cartesian(x_int_backmapped).numpy()

        # endregion

        # region Prepare figure

        if not "width" in plot_args and not "height" in plot_args:
            fig = plt.figure(
                figsize=(
                    main.utils.matplotlib_helpers.column_width * 0.95,
                    main.utils.matplotlib_helpers.column_width * 1.2,
                ),
            )
        else:
            fig = plt.figure(figsize=(plot_args["width"], plot_args["height"]))

        minx = -1.7  # -2.35
        maxx = 1.1
        miny = -0.4
        maxy = 2.1

        ##### Prepare the other axes

        ax_F = plt.gca()

        ax_F.set_xlim(minx, maxx)
        ax_F.set_ylim(miny, maxy)

        divider = make_axes_locatable(ax_F)
        axHist = divider.append_axes("bottom", "30%", pad="12%")
        ax_F_colorbar = divider.append_axes("top", "7%", pad="7%")

        ax_F.axis("equal")
        ax_F.set_xlim(minx, maxx)
        ax_F.set_ylim(miny, maxy)

        # endregion

        # region Plot backmapped free energy profile + training points

        plot_free_energy_histogram(
            x_backmapped[:, 0],
            x_backmapped[:, 1],
            nbins=free_energy_nbins,
            cbar_label="Potential energy / $k_{\mathrm{B}} T$"
            if wandb.config.get("beta") is not None
            else "free energy / kT",
            ax=ax_F,
            cax=ax_F_colorbar,
            cbar_orientation="horizontal",
            vmin=0.0,
            vmax=15.0,
        )

        ax_F.set_xlabel("$x_1$")
        ax_F.set_ylabel("$x_2$")

        ax_F_colorbar.xaxis.set_ticks_position("top")
        ax_F_colorbar.xaxis.set_label_position("top")

        # Plot the so far used training points also on the free energy plot in 2D:
        if training_points is not None and plot_args.get(
            "do_plot_2D_training_points", False
        ):
            ax_F.scatter(
                training_points[:, 0],
                training_points[:, 1],
                s=1,
                facecolors="white",
                edgecolors="black",
                alpha=0.2,
                # marker=".",
            )

        # endregion

        # region Plot the contour lines of the ground truth MB potential

        # Plot the contour lines of the ground truth MB potential
        grid_width = max(maxx - minx, maxy - miny) / 630.0
        xx, yy = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
        # V = muller_potential_regularized(xx, yy)
        V = muller_potential(xx, yy)
        ax_F.contour(
            xx,
            yy,
            V.clip(max=40),
            15,
            cmap="gray",
            linewidths=1.0
            if not "countour_linewidths" in plot_args
            else plot_args["countour_linewidths"],
        )

        # endregion

        # region Plot s CG axis

        ref_point = np.array([-2.25, 1.9])
        direction = np.array([1, -1])
        start_point = ref_point - 1 * direction
        end_point = ref_point + 3 * direction

        ax_F.plot(
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

            ax_F.plot(
                [point[0] - 0.03, point[0] + 0.03],
                [point[1] - 0.03, point[1] + 0.03],
                color="blue",
            )

            ax_F.annotate(
                "${:.1f}$".format(s),
                xy=point,
                xytext=(-16 + (3 if s >= 0 else 0), -7 + (2 if s >= 0 else 0)),
                # xytext=(-26, -16),
                textcoords="offset points",
                color="blue",
                fontsize=7,
            )

        plot_tick(0.0)
        plot_tick(-1.0)
        plot_tick(-2.0)

        ax_F.annotate(
            "s",
            xy=[-0.60, -0.5],
            xytext=(-28, 12),
            textcoords="offset points",
            color="blue",
            fontsize=7,
        )

        # endregion

        # region Plot the mean value for each s in a range of s values

        if plot_means:
            # Ground truth:
            s_values = np.linspace(-2.5, 1.1, 100)
            mean_xs, mean_ys = _get_ground_truth_expected_values_orthogonal_to_s_values(
                s_values
            )
            ax_F.plot(mean_xs, mean_ys, "k", lw=1.5)

            # From the model:
            # Make a tensor from s_values
            s_values_tensor = torch.tensor(s_values, dtype=torch.get_default_dtype())[
                :, None
            ].to(device="cuda")
            means = _get_model_expected_values_orthogonal_to_s_values(
                s_values_tensor, cond_flow
            )
            ax_F.plot(means[:, 0].cpu(), means[:, 1].cpu(), "red", lw=1.5)

            lower_bound, upper_bound = get_model_stdv_orth_to_s(
                s_values_tensor, cond_flow
            )

            ax_F.plot(
                lower_bound[:, 0].cpu(),
                lower_bound[:, 1].cpu(),
                "red",
                lw=1.5,
                linestyle="dashed",
            )
            ax_F.plot(
                upper_bound[:, 0].cpu(),
                upper_bound[:, 1].cpu(),
                "red",
                lw=1.5,
                linestyle="dashed",
            )

        # endregion

        axHist.set_xlim(-2.5, 1.1)

        # region Plot the ground truth free energy profile and the one from the ensemble

        if training_points is not None:
            training_ys = (
                training_points[:, 0] - training_points[:, 1]
            )  # calculate s value

        if free_energy_ensemble is not None:
            free_energy_values_predicted -= np.min(
                free_energy_values_predicted[
                    (s_values_predicted > training_ys.min())
                    & (s_values_predicted < training_ys.max())
                ]
            )

            if not plot_args.get("plot_ensemble_individually", False):
                axHist.plot(
                    s_values_predicted,
                    free_energy_values_predicted,
                    label="Predicted",
                    lw=1.0,
                    color="red",
                )
                axHist.fill_between(
                    s_values_predicted,
                    free_energy_values_predicted - std_dev_predicted,
                    free_energy_values_predicted + std_dev_predicted,
                    color="gray",
                    alpha=0.5,
                )
            else:  # Plot each curve individually
                for i in range(predicted_outputs.shape[0]):
                    current_F = predicted_outputs[i, :].detach().cpu().numpy()
                    current_F -= np.min(
                        current_F[
                            (s_values_predicted > training_ys.min())
                            & (s_values_predicted < training_ys.max())
                        ]
                    )

                    axHist.plot(
                        s_values_predicted,
                        current_F,
                        lw=0.3,
                        color="red",
                        alpha=0.2,
                    )

        ground_truth_free_energies = self.ground_truth_free_energies - np.min(
            self.ground_truth_free_energies
        )
        axHist.plot(
            self.ground_truth_CG_histogram_bin_centers[0],
            ground_truth_free_energies,
            label="Ground truth",
            lw=1.0,
            color="black",
            linestyle="dashed",
        )
        axHist.set_xlim(-2.5, 1.1)

        ymax = np.max(self.ground_truth_free_energies) + 2.0
        axHist.set_ylim(np.min(self.ground_truth_free_energies) - 3.0, ymax)

        # endregion

        # custom_marker = mpath.Path(np.array([[0, -0.3], [0, 0.3]]))

        # Indicate the old points in the histogram
        if training_points is not None:
            axHist.scatter(
                training_ys,
                np.ones_like(training_ys) * ymax - 1.0,
                s=0.05,
                facecolors="black",
                edgecolors="black",
                marker="x",
            )

        # Indicate the new points in the histogram
        if new_points is not None:
            axHist.scatter(
                new_points,
                np.ones_like(new_points) * ymax - 2.0,
                s=0.05,
                facecolors="red",
                edgecolors="red",
                marker="x",
            )

        # Plot the MC starting points
        if MC_starting_points is not None and plot_args.get(
            "plot_MC_starting_points", True
        ):
            axHist.scatter(
                MC_starting_points,
                np.ones_like(MC_starting_points) * ymax - 3.0,
                s=0.05,
                facecolors="blue",
                edgecolors="blue",
                marker="x",
            )

        axHist.set_xlabel("$s=x_1 - x_2$")
        axHist.set_ylabel("Free energy / $k_{\mathrm{B}} T$")

        axHist.legend(loc="lower center")

        set_content_to_rasterized(fig)
        fig.tight_layout()

        figure_name = "backmapping" + (f"_{tag}" if tag is not None else "")
        if output_dir is not None:
            fig.savefig(
                os.path.join(output_dir, figure_name + ".pdf"),
                dpi=600,
                bbox_inches="tight",
            )
        else:  # Log to wandb
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            buf.seek(0)
            wandb.log(
                {
                    figure_name: wandb.Image(Image.open(buf)),
                },
                step=epoch,
            )

        if (
            plot_args.get("plot_std_separately", True)
            and free_energy_ensemble is not None
        ):
            # We only support wandb logging for this

            fig = plt.figure()

            plt.plot(s_values_predicted, std_dev_predicted, color="red", lw=1.0)

            # Also indicate the training points in the plot
            if training_points is not None:
                plt.scatter(
                    training_ys,
                    np.ones_like(training_ys)
                    * wandb.config.MC_sampling["error_threshold"],
                    s=1,
                    c="black",
                    alpha=0.5,
                )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=600)
            buf.seek(0)
            wandb.log(
                {
                    "std": wandb.Image(Image.open(buf)),
                },
                step=epoch,
            )
