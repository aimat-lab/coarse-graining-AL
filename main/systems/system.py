import torch
from typing import List
from torch.autograd import grad
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
import numpy as np
import matplotlib.pyplot as plt


class System:
    def __init__(
        self,
        internal_coordinates_DOF: int,
        CG_indices: List[int],
        ground_truth_trajectory=None,
    ):
        self.internal_coordinates_DOF = internal_coordinates_DOF
        self.CG_indices = CG_indices

        FG_mask = torch.ones(internal_coordinates_DOF, dtype=torch.bool)
        for index in CG_indices:
            FG_mask[index] = False

        self.FG_mask = FG_mask
        self.CG_mask = ~FG_mask

        if ground_truth_trajectory is not None:
            # Process in batches of 1e6 to avoid memory overflow
            ground_truth_trajectory_batches = torch.split(
                ground_truth_trajectory, int(1e6)
            )
            ground_truth_trajectory_batches = [
                self.cartesian_to_internal(batch)
                for batch in ground_truth_trajectory_batches
            ]
            ground_truth_trajectory = torch.cat(ground_truth_trajectory_batches, dim=0)

        self.ground_truth_trajectory_CG = ground_truth_trajectory[:, self.CG_mask]

        self._calculate_histogram_from_ground_truth()

    def get_FG_coordinates(self, x_int: torch.Tensor):
        return x_int[:, self.FG_mask]

    def get_CG_coordinates(self, x_int: torch.Tensor):
        return x_int[:, self.CG_mask]

    def cartesian_to_internal(self, x_cart: torch.Tensor, output_log_det=False):
        raise NotImplementedError("cartesian_to_internal not implemented")

    def internal_to_cartesian(self, x_int: torch.Tensor, output_log_det=False):
        raise NotImplementedError("internal_to_cartesian not implemented")

    def target_log_prob(self, x_int: torch.Tensor):
        raise NotImplementedError("target_log_prob not implemented")

    def run_analysis(self):
        raise NotImplementedError("run_analysis not implemented")

    def get_uniform_CG_ranges(self):
        raise NotImplementedError("get_uniform_range not implemented")

    def filter_chirality(self, x_int):
        raise NotImplementedError("filter_chirality not implemented")

    def target_log_prob_and_grad(self, x_int: torch.Tensor):
        x_int.requires_grad_(True)
        log_prob = self.target_log_prob(x_int)

        gradient = grad(
            log_prob, x_int, grad_outputs=torch.ones_like(log_prob), create_graph=True
        )[0]

        x_int.requires_grad_(False)

        return log_prob, gradient

        """This code seems to not work with custom pytorch autograd functions
        def helper_fn(x):
            potential = self.target_log_prob(x[None, :])
            return potential, potential

        # compute sample-wise gradients of the free energy:
        grad, potential = functorch.vmap(functorch.grad(helper_fn, has_aux=True))(x_int)

        return potential, grad
        """

    def _calculate_histogram_from_ground_truth(self):
        uniform_ranges = self.get_uniform_CG_ranges()

        # Calculate the histogram of the ground truth CG trajectory

        # Convert to numpy
        ground_truth_trajectory_CG_np = self.ground_truth_trajectory_CG.numpy()

        bins_per_dimension = 100

        bins = [
            np.linspace(start, end, num=bins_per_dimension + 1)
            for (start, end) in uniform_ranges
        ]
        histogram, edges = np.histogramdd(ground_truth_trajectory_CG_np, bins=bins)

        histogram_sum = np.sum(histogram)
        if histogram_sum > 0.0:
            # Convert to normalized probabilities over the bins
            histogram = histogram / histogram_sum

        self.ground_truth_CG_probabilities = histogram

        self.ground_truth_free_energies = -np.log(histogram)
        self.ground_truth_free_energies -= np.min(self.ground_truth_free_energies)

        # Determine bin centers
        bin_centers = [0.5 * (edges[i][1:] + edges[i][:-1]) for i in range(len(bins))]
        self.ground_truth_CG_histogram_bin_centers = bin_centers

    def calculate_metrics_to_ground_truth(
        self, free_energy_ensemble: FreeEnergyEnsemble
    ):
        # Create the N-dimensional grid
        mesh = np.meshgrid(*(self.ground_truth_CG_histogram_bin_centers), indexing="ij")
        # Convert the N-dimensional grid to a 2D array
        grid = np.stack(mesh, axis=-1).reshape(-1, len(mesh))

        with torch.no_grad():
            # Determine free energy at the bin centers
            free_energy_at_bin_centers = free_energy_ensemble.mean(
                torch.from_numpy(grid).to(dtype=torch.get_default_dtype()).cuda()
            )

        # Avoid overflow:
        free_energy_at_bin_centers -= torch.median(free_energy_at_bin_centers)

        # Convert to probabilities
        probabilities_at_bin_centers = torch.exp(-free_energy_at_bin_centers)

        ground_truth_CG_histogram_flat = self.ground_truth_CG_probabilities.flatten()

        # Normalize over the bins
        probabilities_at_bin_centers = probabilities_at_bin_centers / torch.sum(
            probabilities_at_bin_centers
        )

        probabilities_at_bin_centers = probabilities_at_bin_centers.cpu().numpy()

        predicted_Fs = -np.log(
            probabilities_at_bin_centers.reshape(
                self.ground_truth_CG_probabilities.shape
            )
        )
        predicted_Fs -= np.min(predicted_Fs)
        ground_truth_Fs = self.ground_truth_free_energies

        predicted_Fs_flat = predicted_Fs.flatten()
        ground_truth_Fs_flat = ground_truth_Fs.flatten()

        if False:
            # Just for testing, plot the free energies at bin centers and the ground truth free energies next to each other

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            predicted_Fs -= np.min(predicted_Fs)

            # Use imshow to show the 2D array as an image
            # In imshow, the first index corresponds to the height of the image, so we have to transpose Fs
            ax[0].imshow(
                predicted_Fs.T,
                extent=[-np.pi, np.pi, -np.pi, np.pi],
                origin="lower",
                cmap="nipy_spectral",
                interpolation="nearest",
                aspect="auto",
                vmax=11.0,
                vmin=0.0,
            )
            ax[0].set_title("Predicted")

            ground_truth_Fs -= np.min(ground_truth_Fs)

            ax[1].imshow(
                ground_truth_Fs.T,
                extent=[-np.pi, np.pi, -np.pi, np.pi],
                origin="lower",
                cmap="nipy_spectral",
                interpolation="nearest",
                aspect="auto",
                vmax=11.0,
                vmin=0.0,
            )
            ax[1].set_title("Ground truth")

            plt.show()

        epsilon = 1e-10
        mask = ground_truth_CG_histogram_flat > 0.0

        # Calculate KL divergence between ground truth and free energy ensemble
        forward_KL_divergence = np.sum(
            ground_truth_CG_histogram_flat[mask]
            * np.log(
                ground_truth_CG_histogram_flat[mask]
                / probabilities_at_bin_centers[mask]
            )
        )  # epsilon not needed here because probabilities_at_bin_centers is never exactly zero

        reverse_KL_divergence = np.sum(
            probabilities_at_bin_centers
            * np.log(
                (probabilities_at_bin_centers + epsilon)
                / (ground_truth_CG_histogram_flat + epsilon)
            )
        )

        MSE_prob = np.mean(
            (ground_truth_CG_histogram_flat - probabilities_at_bin_centers) ** 2
        )
        MSE_free_energy = np.mean(
            (ground_truth_Fs_flat[mask] - predicted_Fs_flat[mask]) ** 2
        )

        return {
            "forward_KL_divergence": forward_KL_divergence,
            "reverse_KL_divergence": reverse_KL_divergence,
            "MSE_prob": MSE_prob,
            "MSE_free_energy": MSE_free_energy,
        }
