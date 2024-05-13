from main.active_learning.samplers.active_learning_sampler import ActiveLearningSampler
import torch
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
from typing import Tuple
import math
import wandb
import os
import time
import logging


class ActiveLearningSamplerMC(ActiveLearningSampler):
    def __init__(
        self,
        ensemble: FreeEnergyEnsemble,
        step_size: float = 0.1,
        error_threshold: float = 1.0,
        periodic_wrapping_ranges: torch.Tensor = None,
    ):
        """This sampler is used to sample new high-error points from the ensemble using MC.

        Args:
            ensemble (FreeEnergyEnsemble): The ensemble to sample from. The energies of the ensemble
                should be in units of kT (1/beta).
            step_size (float, optional): The step size to use for the Metropolis-Hastings algorithm.
                Defaults to 0.1.
            error_threshold (float, optional): The threshold for the error. If the error exceeds this
                threshold, the trajectory is terminated. Defaults to 1.0.
            periodic_wrapping_ranges (torch.Tensor, optional): The ranges to use for periodic wrapping
                for each dimension. Shape: (D, 2). Defaults to None.
        """

        super().__init__(ensemble)
        self.step_size = step_size
        self.error_threshold = error_threshold
        self.periodic_wrapping_ranges = periodic_wrapping_ranges

        self.max_time = wandb.config["MC_sampling"].get("max_time")
        self.max_iterations = wandb.config["MC_sampling"].get("max_iterations")

    def _metropolis_criterium(
        self, current_energies: torch.Tensor, proposed_energies: torch.Tensor
    ) -> torch.Tensor:
        acceptance_probabilities = torch.exp(
            -1.0 * (proposed_energies - current_energies)
        )
        return torch.rand_like(proposed_energies) < acceptance_probabilities

    def _propose_new_points(self, current_points: torch.Tensor) -> torch.Tensor:
        return current_points + torch.randn_like(current_points) * self.step_size

    def sample_new_higherror_points(
        self,
        starting_points: torch.Tensor,
        NO_trajectories: int = 10,
        NO_points_to_find: int = 50,
        debugging: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Samples new high-error points from the D-dimensional ensemble using the Metropolis-Hastings algorithm.

        Args:
            starting_points (torch.Tensor): Tensor of shape (N,D) containing the starting points.
            NO_trajectories (int, optional): The number of trajectories to run in parallel.
                If NO_trajectories < starting_points.shape[0], we will take the first
                NO_trajectories starting points. If NO_trajectories > starting_points.shape[0],
                we will repeat the starting points. Defaults to 100.
            NO_points_to_find (int, optional): NO_points_to_find <= NO_trajectories. The number of
                high-error points to find. Defaults to 50.
            debugging (bool, optional): Whether to return the trajectories instead. Defaults to False.

        Returns:
            torch.Tensor: Tensor of shape (N,D) containing the found high-error points.
            int: The total number of steps taken by the sampler.
        """

        start_time = time.time()

        if not starting_points.is_cuda:
            starting_points = starting_points.cuda()

        with torch.no_grad():
            if NO_trajectories <= starting_points.shape[0]:
                points = starting_points[0:NO_trajectories]
            else:
                points = starting_points.repeat(
                    (math.ceil(NO_trajectories / starting_points.shape[0]), 1)
                )[0:NO_trajectories]

            energies, std_devs = self.ensemble.mean_and_std_dev(points)

            # Array to track whether the error for each trajectory has exceeded the threshold
            completed = torch.full((points.shape[0],), False, device="cuda")
            length_of_trajectories = torch.zeros(
                (points.shape[0],), device="cuda", dtype=torch.int32
            )

            step_counter = 0

            if debugging:
                trajectories = torch.zeros((points.shape[0], 1000000))

            accept_fullsize = torch.full((points.shape[0],), False, device="cuda")

            if self.periodic_wrapping_ranges is not None:
                if not self.periodic_wrapping_ranges.is_cuda:
                    periodic_wrapping_ranges = self.periodic_wrapping_ranges.cuda()
                else:
                    periodic_wrapping_ranges = self.periodic_wrapping_ranges

                mins = periodic_wrapping_ranges[:, 0].unsqueeze(0)
                maxs = periodic_wrapping_ranges[:, 1].unsqueeze(0)
                ranges = maxs - mins

            while not (torch.sum(completed) >= NO_points_to_find):
                if debugging and step_counter % 1000 == 0:
                    print(step_counter)

                proposed_points = self._propose_new_points(points[~completed])

                if self.periodic_wrapping_ranges is not None:
                    proposed_points = (proposed_points - mins) % ranges + mins

                proposed_energies, proposed_std_devs = self.ensemble.mean_and_std_dev(
                    proposed_points
                )

                accept_fullsize[:] = False

                accept = self._metropolis_criterium(
                    energies[~completed], proposed_energies
                )
                accept_fullsize[~completed] = accept

                points[accept_fullsize] = proposed_points[accept]
                energies[accept_fullsize] = proposed_energies[accept]
                std_devs[accept_fullsize] = proposed_std_devs[accept]

                if debugging:
                    trajectories[:, step_counter] = points.cpu()

                # Noting the length of trajectories
                length_of_trajectories[~completed] += 1

                error_too_high = std_devs[~completed] > self.error_threshold
                completed[~completed] = error_too_high & (
                    length_of_trajectories[~completed]
                    > wandb.config.min_NO_steps_MC_trajectories
                )

                step_counter += 1

                if (step_counter % 100000) == 0:
                    torch.cuda.empty_cache()  # avoid OOM

                if (step_counter % 100) == 0:
                    if self.max_time is not None:
                        if (time.time() - start_time) > self.max_time:
                            logging.info(
                                "Time limit of MC sampling reached. Stopping AL workflow..."
                            )
                            return (
                                None,
                                torch.sum(length_of_trajectories).item(),
                                step_counter,
                            )

                if debugging:
                    if step_counter >= 1000000:
                        return trajectories

                if self.max_iterations is not None:
                    if step_counter >= self.max_iterations:
                        logging.info(
                            "Maximum number of iterations reached. Stopping AL workflow..."
                        )
                        return (
                            None,
                            torch.sum(length_of_trajectories).item(),
                            step_counter,
                        )

        return (
            points[completed][0:NO_points_to_find],
            torch.sum(length_of_trajectories).item(),
            step_counter,
        )
