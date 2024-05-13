from main.systems.system import System
from typing import List
import torch
from main.systems.aldp.target_distribution import prepare_target
import wandb
import boltzgen as bg
import numpy as np
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
import os
from main.systems.aldp.coord_trafo import unscale_internal_coordinates
from main.systems.aldp.coord_trafo import scale_internal_coordinates
from main.systems.aldp.coord_trafo import get_means_and_stds
import io
import PIL.Image as Image
import copy
from main.utils.call_model_batched import call_model_batched
import matplotlib.pyplot as plt
from main.systems.aldp.coord_trafo import get_dih_mean_scale
from main.utils.matplotlib_helpers import set_content_to_rasterized


def get_uniformly_distributed_dihedrals_scaled(trafo, random=True, N=1000):
    if random:
        phi = 2 * torch.pi * torch.rand(N * N) - torch.pi
        psi = 2 * torch.pi * torch.rand(N * N) - torch.pi
    else:
        # Uniformly distributed on a grid
        phi = torch.linspace(-np.pi, np.pi, N)
        psi = torch.linspace(-np.pi, np.pi, N)

        phi, psi = torch.meshgrid(phi, psi)

        phi = phi.reshape(-1)
        psi = psi.reshape(-1)

    mean_phi, scale_phi = get_dih_mean_scale(wandb.config.CG_indices[0], trafo)
    mean_psi, scale_psi = get_dih_mean_scale(wandb.config.CG_indices[1], trafo)

    # Watch out the different sign conventions!
    phi_scaled = -1.0 * phi - mean_phi
    # Fix dihedral
    phi_scaled = (phi_scaled + torch.pi) % (2 * torch.pi) - torch.pi
    phi_scaled /= scale_phi

    psi_scaled = -1.0 * psi - mean_psi
    # Fix dihedral
    psi_scaled = (psi_scaled + torch.pi) % (2 * torch.pi) - torch.pi
    psi_scaled /= scale_psi

    CG_configurations = torch.stack([phi_scaled, psi_scaled], dim=1)

    # Convert to degrees
    phi_dih = torch.rad2deg(phi)
    psi_dih = torch.rad2deg(psi)

    return CG_configurations, phi_dih, psi_dih


class AldpSystem(System):
    def __init__(
        self,
        internal_coordinates_DOF: int,
        CG_indices: List[int],
        trafo: bg.flows.CoordinateTransform,
        ground_truth_trajectory=None,
    ):
        self.trafo = copy.deepcopy(trafo).to("cuda")
        self.trafo_cpu = copy.deepcopy(trafo).to("cpu")

        self.target = prepare_target(
            self.trafo_cpu
        )  # For now, we are using CPU for all target calls
        # This means that all trafos to cartesian coordinates are executed on the CPU, but this should be fine.

        self._means_and_stds_cpu = get_means_and_stds(self.trafo_cpu)
        self._means_and_stds = [item.cuda() for item in self._means_and_stds_cpu]

        super().__init__(internal_coordinates_DOF, CG_indices, ground_truth_trajectory)

    def cartesian_to_internal(self, x_cart: torch.Tensor, output_log_det=False):
        if x_cart.is_cuda:
            output = self.trafo.transform.forward(x_cart)
        else:
            output = self.trafo_cpu.transform.forward(x_cart)

        if output_log_det:
            return output
        else:
            return output[0]

    def internal_to_cartesian(self, x_int: torch.Tensor, output_log_det=False):
        if x_int.is_cuda:
            output = self.trafo.transform.inverse(x_int)
        else:
            output = self.trafo_cpu.transform.inverse(x_int)

        if output_log_det:
            return output
        else:
            return output[0]

    def convert_to_unscaled_internal(
        self, x_int: torch.Tensor, indices: List[int] = None
    ):
        return unscale_internal_coordinates(
            x_int,
            self.trafo if x_int.is_cuda else self.trafo_cpu,
            self._means_and_stds if x_int.is_cuda else self._means_and_stds_cpu,
            indices=indices,
        )

    def convert_to_scaled_internal(
        self, x_int_unscaled: torch.Tensor, indices: List[int] = None
    ):
        return scale_internal_coordinates(
            x_int_unscaled,
            self.trafo if x_int_unscaled.is_cuda else self.trafo_cpu,
            self._means_and_stds
            if x_int_unscaled.is_cuda
            else self._means_and_stds_cpu,
            indices=indices,
        )

    def invert_cg_configurations(self, x_cg: torch.Tensor):
        x_cg_unscaled = self.convert_to_unscaled_internal(x_cg, indices=self.CG_indices)
        x_cg_unscaled *= -1.0
        x_cg = self.convert_to_scaled_internal(x_cg_unscaled, indices=self.CG_indices)

        return x_cg

    def target_log_prob(self, x_int: torch.Tensor):
        if x_int.is_cuda:
            x_int_cpu = x_int.cpu()
        else:
            x_int_cpu = x_int

        output = self.target.log_prob(x_int_cpu)

        if x_int.is_cuda:
            return output.cuda()
        else:
            return output

    def get_uniform_CG_ranges(self):
        ranges = [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
        ]
        ranges = torch.Tensor(ranges)
        return ranges

    def wrap_cg_coordinates(self, x_cg: torch.Tensor):
        x_cg = (x_cg + np.pi) % (2 * np.pi) - np.pi
        return x_cg

    def filter_chirality(
        self, x_int, ind=[17, 26], mean_diff=-0.043, threshold=0.8, output_float=False
    ):
        """Filters batch for the L-form. Code from Midgley et al.

        Args:
            x (torch.Tensor): Input batch
            ind (List): Indices to be used for determining the chirality
            mean_diff (float): Mean of the difference of the coordinates
            threshold (float): Threshold to be used for splitting
            output_float (bool): If True, additionally returns the float

        Returns:
            torch.Tensor: Returns mask of batch, where L-form is present
        """

        diff_ = torch.column_stack(
            (
                x_int[:, ind[0]] - x_int[:, ind[1]],
                x_int[:, ind[0]] - x_int[:, ind[1]] + 2 * np.pi,
                x_int[:, ind[0]] - x_int[:, ind[1]] - 2 * np.pi,
            )
        )
        min_diff_ind = torch.min(torch.abs(diff_), 1).indices
        diff = diff_[torch.arange(x_int.shape[0]), min_diff_ind]

        float_value = torch.abs(diff - mean_diff)
        ind = float_value < threshold

        if output_float:
            return ind, float_value
        else:
            return ind

    def run_analysis(
        self,
        epoch: int,
        free_energy_ensemble: FreeEnergyEnsemble = None,
        training_points: torch.Tensor = None,
        new_points: torch.Tensor = None,
        MC_starting_points: torch.Tensor = None,
        new_points_training_centers: torch.Tensor = None,
        output_dir: str = None,
        tag: str = None,
        plot_args: dict = {},
    ):
        """Run analysis.

        Args:
            epoch (int): Epoch number.
            free_energy_ensemble (FreeEnergyEnsemble, optional): Free energy ensemble. Defaults to None.
            training_points (torch.Tensor, optional): Points in the AL dataset so far. Defaults to None.
            new_points (torch.Tensor, optional): New s-points that have just been added to the training set.
                Defaults to None.
            MC_starting_points (torch.Tensor, optional): Starting points for the MC sampling. Defaults to None.
            new_points_training_centers (torch.Tensor, optional): Centers of the new points that have just been added to the training set.
                Defaults to None.
            output_dir (str, optional): Path to save the figure to. If this is None, the figure is logged to wandb.
                Defaults to None.
            tag (str, optional): Tag to add to the name for logging / saving the figure. Defaults to None.
            plot_args (dict, optional): Additional arguments for plotting. Defaults to None.
        """

        (
            CG_configurations,
            phi_dih,
            psi_dih,
        ) = get_uniformly_distributed_dihedrals_scaled(
            self.trafo_cpu, random=False, N=1000
        )

        with torch.no_grad():
            # Returns a tuple of (samples_backmapped [N,1], jacobian [N])
            free_energies, std_dev = call_model_batched(
                free_energy_ensemble.mean_and_std_dev,
                CG_configurations.detach(),
                device="cuda",
                batch_size=4096,
                droplast=False,
                do_detach=False,
                move_back_to_cpu=True,
            )

        free_energies -= torch.min(free_energies)

        fig_free_energies = plt.figure()
        plt.imshow(
            free_energies.reshape(1000, 1000).numpy().T,
            extent=[
                -180.0,
                180.0,
                -180.0,
                180.0,
            ],
            origin="lower",
            cmap="nipy_spectral",
            interpolation="nearest",
            aspect="auto",
            vmin=0.0,
            vmax=25.0,
        )
        # print("Free_energy_max:", torch.max(free_energies))

        if ("disable_colorbar" not in plot_args) or not plot_args["disable_colorbar"]:
            plt.colorbar(label="Free energy / $k_{\mathrm{B}} T$")
        if ("disable_labels" not in plot_args) or not plot_args["disable_labels"]:
            plt.xlabel(r"$\phi$")
            plt.ylabel(r"$\psi$")

        plt.xticks([-90, 0, 90])
        plt.yticks([-90, 0, 90])

        fig_std_dev = plt.figure()
        plt.imshow(
            std_dev.reshape(1000, 1000).numpy().T,
            extent=[
                -180.0,
                180.0,
                -180.0,
                180.0,
            ],
            origin="lower",
            cmap="nipy_spectral",
            interpolation="nearest",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        if ("disable_colorbar" not in plot_args) or not plot_args["disable_colorbar"]:
            plt.colorbar(label="Standard deviation / $k_{\mathrm{B}} T$")

        plt.contour(
            std_dev.reshape(1000, 1000).numpy().T,
            extent=[
                -180.0,
                180.0,
                -180.0,
                180.0,
            ],
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            levels=[wandb.config.MC_sampling["error_threshold"]],
            colors=["black"],
            zorder=10,
        )

        if ("disable_labels" not in plot_args) or not plot_args["disable_labels"]:
            plt.xlabel(r"$\phi$")
            plt.ylabel(r"$\psi$")

        plt.xticks([-90, 0, 90])
        plt.yticks([-90, 0, 90])

        # Plot training points
        if training_points is not None and plot_args.get(
            "display_training_points", True
        ):
            CG_unscaled = self.convert_to_unscaled_internal(
                training_points[:, self.CG_mask], indices=self.CG_indices
            )
            CG_unscaled *= -1.0  # Sign convention
            CG_unscaled = CG_unscaled.numpy()

            # Convert from radian to degree
            CG_unscaled *= 360.0 / (2 * np.pi)

            for axis in [fig_free_energies.gca(), fig_std_dev.gca()]:
                axis.scatter(
                    CG_unscaled[:, 0],
                    CG_unscaled[:, 1],
                    c="black",
                    edgecolors="none",
                    alpha=1.0,  # 0.2 / wandb.config.number_of_z_per_y,
                    s=0.1,
                    zorder=11,
                )

        # Plot new points sampled from the current free energy ensemble
        if new_points is not None and not (
            "display_new_points" in plot_args and not plot_args["display_new_points"]
        ):
            CG_unscaled = self.convert_to_unscaled_internal(
                new_points, indices=self.CG_indices
            )
            CG_unscaled *= -1.0
            CG_unscaled = CG_unscaled.numpy()

            # Convert from radian to degree
            CG_unscaled *= 360.0 / (2 * np.pi)

            for axis in [fig_free_energies.gca(), fig_std_dev.gca()]:
                axis.scatter(
                    CG_unscaled[:, 0],
                    CG_unscaled[:, 1],
                    c="red",
                    edgecolors="none",
                    alpha=1.0,  # 0.2 / wandb.config.number_of_z_per_y,
                    s=0.1,
                    zorder=12,
                )

        if MC_starting_points is not None and plot_args.get(
            "plot_MC_starting_points", True
        ):
            CG_unscaled = self.convert_to_unscaled_internal(
                MC_starting_points, indices=self.CG_indices
            )
            CG_unscaled *= -1.0
            CG_unscaled = CG_unscaled.numpy()

            # Convert from radian to degree
            CG_unscaled *= 360.0 / (2 * np.pi)

            for axis in [fig_free_energies.gca(), fig_std_dev.gca()]:
                axis.scatter(
                    CG_unscaled[:, 0],
                    CG_unscaled[:, 1],
                    c="white",
                    edgecolors="none",
                    alpha=1.0,
                    s=2.5,
                    zorder=13,
                )

        """
        if new_points_training_centers is not None:
            CG_unscaled = self.convert_to_unscaled_internal(
                new_points_training_centers, indices=self.CG_indices
            )
            CG_unscaled *= -1.0
            CG_unscaled = CG_unscaled.numpy()

            # Convert from radian to degree
            CG_unscaled *= 360.0 / (2 * np.pi)

            for axis in [fig_free_energies.gca(), fig_std_dev.gca()]:
                axis.scatter(
                    CG_unscaled[:, 0],
                    CG_unscaled[:, 1],
                    c="blue",
                    edgecolors="none",
                    alpha=1.0,
                    s=1.0,
                    zorder=13,
                )
        """

        for fig, name in zip(
            [fig_free_energies, fig_std_dev],
            ["free_energies", "std_dev"],
        ):
            if tag is not None:
                name += f"_{tag}"

            if output_dir is not None:
                set_content_to_rasterized(fig)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(output_dir, f"{name}.pdf"),
                    dpi=600,
                    bbox_inches="tight",
                )
            else:  # Log to wandb
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                buf.seek(0)
                wandb.log(
                    {
                        name: wandb.Image(Image.open(buf)),
                    },
                    step=epoch,
                )
