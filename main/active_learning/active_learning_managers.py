import wandb
from main.utils.training_utils import train_single_batch_by_example
from main.utils.training_utils import train_epoch_reweighted_by_probability
import math
import torch
from main.active_learning.samplers.active_learning_sampler import ActiveLearningSampler
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
from main.active_learning.active_learning_dataset import ActiveLearningDataset
from main.utils.training_utils import calculate_test_KL_by_probability_loss
from main.utils.FastTensorDataLoader import FastTensorDataLoader
import os
from main.systems.system import System
import time
import logging
from main.utils.newline_tqdm import NewlineTqdm as tqdm
from torch.optim.lr_scheduler import LambdaLR
from main.models.flow_base import ConditionalFlowBase
from main.systems.aldp.coord_trafo import alternative_filter_chirality

# import yappi


def generate_grid(uniform_ranges, num_points_per_dim):
    """
    Generate an N-dimensional grid of equidistant points.

    Args:
        uniform_ranges (list of tuples or Tensor of shape (D,2)): List where each tuple is a (min, max) for that dimension.
        num_points_per_dim (int): Number of equidistant points per dimension.

    Returns:
        grid (torch.Tensor): A 2D array of shape (M, number_of_dimensions).
    """

    # Create linearly spaced points for each dimension
    axis_coords = [
        torch.linspace(low, high, num_points_per_dim) for low, high in uniform_ranges
    ]

    # Create the N-dimensional grid
    mesh = torch.meshgrid(*axis_coords, indexing="ij")

    # Convert the N-dimensional grid to a 2D array
    grid = torch.stack(mesh, axis=-1).reshape(-1, len(uniform_ranges))

    return grid


def select_diverse_points(tensor, N):
    # Calculate pairwise distances
    dist_matrix = torch.cdist(tensor, tensor, p=2)

    # Start with a random point
    selected_indices = [torch.randint(0, tensor.size(0), (1,)).item()]

    # Iteratively select N-1 more points
    for _ in range(1, N):
        # Minimum distances to the set of selected points
        min_distances = dist_matrix[selected_indices].min(dim=0)[0]

        # Choose the point with the maximum minimum distance
        next_index = min_distances.argmax().item()
        selected_indices.append(next_index)

    # Gather the selected points
    diverse_points = tensor[selected_indices]

    return diverse_points


class EnsembleALManager:
    def __init__(
        self,
        cond_flow: ConditionalFlowBase,
        cond_flow_optimizer: torch.optim.Optimizer,
        system: System,
        AL_ensemble: FreeEnergyEnsemble,
        AL_dataset: ActiveLearningDataset,
        AL_sampler: ActiveLearningSampler,
        starting_dataloader: FastTensorDataLoader,
    ):
        """Base class for active learning managers.

        Args:
            cond_flow: The invertible neural network to be trained.
            cond_flow_optimizer: The optimizer to be used for the invertible model.
            system: The target system.
            AL_ensemble: The ensemble of models used for active learning.
            AL_dataset: The active learning dataset.
            AL_sampler: The sampler to be used for active learning.
            starting_dataloader: The dataloader to be used for the initial epochs trained by example.
        """

        self.cond_flow = cond_flow
        self.cond_flow_optimizer = cond_flow_optimizer
        self.AL_ensemble = AL_ensemble
        self.AL_dataset = AL_dataset
        self.starting_dataloader = starting_dataloader
        self.system = system

        self.total_NO_evaluations = 0
        self.total_NO_evaluations_after_filtering = 0

        self.AL_sampler = AL_sampler
        self.previous_new_points = None
        self.total_NO_MC_steps = 0

        self.current_iteration = 0
        self.current_epoch_in_iteration = (
            0  # Epoch counter within the current iteration
        )
        self.current_epoch = 0  # Global epoch counter
        self.current_progress_bar = tqdm(
            total=EnsembleALManager.get_iteration_length(self.current_iteration),
            desc="Training by example",
            mininterval=30,
        )

        self.lr_scheduler = None  # Will be created after the first iteration

    def initialize_lr_scheduler(self):
        # Define lr lambda for linear warmup in the beginning of each iteration
        def lr_lambda(epoch):
            if self.current_iteration == 0:
                return 1.0  # training by example
            elif self.current_iteration == 1:
                NO_warmup_epochs = wandb.config.get(
                    "warmup_epochs_by_probability_initial_iteration", -1
                )
            else:
                NO_warmup_epochs = wandb.config.get("warmup_epochs_by_probability", -1)

            if self.current_epoch_in_iteration < NO_warmup_epochs:
                return (self.current_epoch_in_iteration + 1) / NO_warmup_epochs
            else:
                if wandb.config.get("lr_decay") is None:
                    return 1.0
                else:
                    if wandb.config["lr_decay"]["type"] == "cosine":
                        return 0.5 * (
                            1
                            + math.cos(
                                math.pi
                                * (
                                    (self.current_epoch_in_iteration - NO_warmup_epochs)
                                    / (
                                        EnsembleALManager.get_iteration_length(
                                            self.current_iteration
                                        )
                                        - NO_warmup_epochs
                                    )
                                )
                            )
                        )

                    elif wandb.config["lr_decay"]["type"] == "steps":
                        current_lambda = 1.0

                        for step, factor in wandb.config["lr_decay"]["steps"]:
                            if self.current_epoch_in_iteration >= step:
                                current_lambda *= factor

                        return current_lambda

        lr_scheduler = LambdaLR(self.cond_flow_optimizer, lr_lambda=lr_lambda)
        self.lr_scheduler = lr_scheduler

    def get_iteration_length(iteration):
        if iteration == 0:
            return wandb.config.N_epochs_by_example
        elif iteration == 1:
            return wandb.config.epochs_in_initial_range
        else:
            return wandb.config.epochs_per_iteration

    def _train_epoch_by_example(self):
        batch_idx = 0

        sum_KL_loss_by_example = 0

        for coords in self.starting_dataloader:
            coords = coords[0]
            coords = coords.to("cuda")

            batch_idx += 1

            self.cond_flow_optimizer.zero_grad()

            l, KL_loss_by_example = train_single_batch_by_example(
                coords,
                cond_flow=self.cond_flow,
                system=self.system,
            )

            sum_KL_loss_by_example += KL_loss_by_example

            l.backward()
            self.cond_flow_optimizer.step()

        wandb.log(
            {
                "total_loss": sum_KL_loss_by_example / batch_idx,
                "KL_loss_by_example": sum_KL_loss_by_example / batch_idx,
                "KL_loss_by_probability": 0.0,
                "sum_of_weights_vector_before_normalization": 0.0,
                "NO_evaluations_per_epoch": 0,
                "NO_evaluations_after_filtering_per_epoch": 0,
                "total_NO_evaluations": 0,
                "total_NO_evaluations_after_filtering": 0,
                "test_KL_by_probability_loss": 0.0,
            },
            step=self.current_epoch,
        )

    def create_checkpoint(
        self,
        new_points_training: torch.Tensor = None,
        new_points_training_group_indices: torch.Tensor = None,
        new_points_test: torch.Tensor = None,
        new_points_test_group_indices: torch.Tensor = None,
        new_points_training_centers: torch.Tensor = None,
        MC_starting_points: torch.Tensor = None,
        silence_stopped_print: bool = False,
        calculate_metrics: bool = True,
        s_samples: torch.Tensor = None,
        plot_predicted_F: bool = True,
        display_training_points: bool = True,
        n_bins_backmapping: int = 100,
        do_not_save: bool = False,
        print_instead_of_logging: bool = False,
        analysis_output_dir: str = None,
        save_checkpoint_dir: str = None,
        tag: str = "",
        plot_args: dict = {},
        force_stop: bool = False,
    ):
        """Creates a checkpoint. First, this saves the state of all models and optimizers.
        Then, this runs the analysis of the current state of the models. This will also determine
        whether the training should be stopped. If the training should be stopped, this function
        will return True, indicating that the training should be stopped.

        Args:
            new_points_training (torch.Tensor, optional): The new points that were just added to the training set of the AL dataset.
                Defaults to None.
            new_points_training_group_indices (torch.Tensor, optional): The group indices of the new points that were just added to the training set of the AL dataset.
                Defaults to None.
            new_points_test (torch.Tensor, optional): The new points that were just added to the test set of the AL dataset.
                Defaults to None.
            new_points_test_group_indices (torch.Tensor, optional): The group indices of the new points that were just added to the test set of the AL dataset.
                Defaults to None.
            new_points_training_centers (torch.Tensor, optional): The centers of the new points that were just added to the AL dataset.
                Defaults to None.
            MC_starting_points (torch.Tensor, optional): The starting points used for the Monte Carlo sampling.
            silence_stopped_print (bool, optional): Whether to silence the print when the training is stopped.
            calculate_metrics (bool, optional): Whether to calculate the metrics between the ground truth and the predicted free energies.
                Defaults to True.
            s_samples (torch.Tensor, optional): The samples used for the backmapping (only needed if system_backmapping_style=="direct").
                If not supplied, samples will be generated using rejection sampling.
            plot_predicted_F (bool, optional): Whether to plot the predicted free energies. Defaults to True.
            display_training_points (bool, optional): Whether to display the training points in the analysis plots.
                Only relevant if system_backmapping_style=="direct".
                Defaults to True.
            n_bins_backmapping (int, optional): The number of bins to use for the backmapping.
                Only relevant if system_backmapping_style=="direct".
                Defaults to 100.
            do_not_save (bool, optional): Whether to not save the checkpoint and just run analysis. Defaults to False.
            print_instead_of_logging (bool, optional): Whether to print metrics instead of logging to wandb. Defaults to False.
            analysis_output_dir (str, optional): The directory to save the analysis plots to. If None, the plots will be logged to wandb.
                Defaults to None.
            save_checkpoint_dir (str, optional): The directory to save the checkpoint to. If None, the checkpoint will be saved to the wandb run directory.
                Defaults to None.
            tag (str, optional): A tag to be added to the analysis plots. Defaults to "".
            plot_args (dict, optional): Additional arguments to be passed to the plotting function. Defaults to {}.
            force_stop (bool, optional): Whether to force the training to stop. Defaults to False.

        Returns:
            bool: True if the training should be stopped, False otherwise.
        """

        # region Save all states

        if not do_not_save:
            if save_checkpoint_dir is None:
                checkpoint_dir = os.path.join(
                    wandb.run.dir, "checkpoints", f"epoch_{self.current_epoch}"
                )
            else:
                checkpoint_dir = save_checkpoint_dir

            os.system("mkdir -p " + checkpoint_dir)

            # This includes the INN state:
            self.AL_dataset.save_state(os.path.join(checkpoint_dir, "AL_dataset.pt"))
            # INN optimizer still missing:
            torch.save(
                self.cond_flow_optimizer.state_dict(),
                os.path.join(checkpoint_dir, "INN_optimizer.pt"),
            )

            # This includes both the models and the optimizers:
            self.AL_ensemble.save_state(os.path.join(checkpoint_dir, "ensemble.pt"))

            torch.save(
                {
                    "new_points_training": new_points_training,
                    "new_points_training_group_indices": new_points_training_group_indices,
                    "new_points_test": new_points_test,
                    "new_points_test_group_indices": new_points_test_group_indices,
                    "new_points_training_centers": new_points_training_centers,
                    "MC_starting_points": MC_starting_points,
                    "display_training_points": display_training_points,
                    "plot_predicted_F": plot_predicted_F,
                    "s_samples": s_samples,
                    "n_bins_backmapping": n_bins_backmapping,
                    "total_NO_evaluations": self.total_NO_evaluations,
                    "total_NO_evaluations_after_filtering": self.total_NO_evaluations_after_filtering,
                    "current_iteration": self.current_iteration,
                    "current_epoch_in_iteration": self.current_epoch_in_iteration,
                    "current_epoch": self.current_epoch,
                    "total_NO_MC_steps": self.total_NO_MC_steps,
                    "previous_new_points": self.previous_new_points,
                },
                os.path.join(checkpoint_dir, "additional.pt"),
            )

        # endregion

        # region Calculate metrics

        if calculate_metrics:
            metrics = self.system.calculate_metrics_to_ground_truth(self.AL_ensemble)

            if not print_instead_of_logging:
                wandb.log(metrics, step=self.current_epoch)
            else:
                logging.info(f"Metrics of epoch {self.current_epoch}:\n{metrics}")

            termination_metric = wandb.config.AL_termination_condition["metric_name"]
            threshold = wandb.config.AL_termination_condition["threshold"]

            stopped = False

            if threshold is not None:
                if metrics[termination_metric] < threshold:
                    if not silence_stopped_print:
                        logging.info(
                            f"Stopping training because {termination_metric} < {threshold}"
                        )
                    stopped = True

                    # Do not display new points in analysis plots for the final checkpoint
                    new_points_training = None
                    new_points_training_centers = None
                    new_points_training_group_indices = None
                    new_points_test = None
                    new_points_test_group_indices = None
        else:
            stopped = False

        if force_stop:
            stopped = True

        # endregion

        # region Run analysis

        if wandb.config.target_system["system_backmapping_style"] == "direct":
            current_ys = self.AL_dataset.get_current_s()
            min_max_s = (torch.min(current_ys).item(), torch.max(current_ys).item())

            for plot_means in [True, False]:
                self.system.run_analysis(
                    cond_flow=self.cond_flow,
                    epoch=self.current_epoch,
                    tag=f"all_{'means' if plot_means else 'no_means'}" + tag,
                    s_samples=s_samples if s_samples is not None else None,
                    training_points=self.system.internal_to_cartesian(
                        self.AL_dataset.get_current_xs_int()
                    ).numpy()
                    if display_training_points
                    else None,
                    new_points=new_points_training.numpy()
                    if ((new_points_training is not None) and not stopped)
                    else None,
                    MC_starting_points=MC_starting_points,
                    plot_means=plot_means,
                    min_max_s=min_max_s,
                    free_energy_ensemble=self.AL_ensemble if plot_predicted_F else None,
                    free_energy_nbins=n_bins_backmapping,
                    output_dir=analysis_output_dir,
                    plot_args=plot_args,
                )
        elif wandb.config.target_system["system_backmapping_style"] == "indirect":
            if plot_predicted_F:
                for display_training_points_local in [True, False]:
                    self.system.run_analysis(
                        self.current_epoch,
                        self.AL_ensemble,
                        output_dir=analysis_output_dir,
                        training_points=(
                            self.AL_dataset.get_current_xs_int()
                            if display_training_points
                            else None
                        )
                        if display_training_points_local
                        else None,
                        new_points=(
                            new_points_training
                            if ((new_points_training is not None) and not stopped)
                            else None
                        )
                        if display_training_points_local
                        else None,
                        MC_starting_points=MC_starting_points,
                        new_points_training_centers=new_points_training_centers
                        if display_training_points_local
                        else None,
                        tag=(
                            "training_points"
                            if display_training_points_local
                            else "no_training_points"
                        )
                        + tag,
                        plot_args=plot_args,
                    )
        else:
            raise ValueError(
                f"Invalid value for system_backmapping_style: {wandb.config.target_system['system_backmapping_style']}"
            )

        # endregion

        return stopped

    def load_checkpoint(self, checkpoint_dir: str):
        """Loads a checkpoint.

        Args:
            checkpoint_dir (str): The directory of the checkpoint.
        """

        if self.AL_dataset is not None:
            self.AL_dataset.load_state(os.path.join(checkpoint_dir, "AL_dataset.pt"))

        if self.cond_flow_optimizer is not None:
            self.cond_flow_optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "INN_optimizer.pt"))
            )

        if self.AL_ensemble is not None:
            self.AL_ensemble.load_state(os.path.join(checkpoint_dir, "ensemble.pt"))

        additional = torch.load(os.path.join(checkpoint_dir, "additional.pt"))
        self.total_NO_evaluations = additional["total_NO_evaluations"]
        self.total_NO_evaluations_after_filtering = additional[
            "total_NO_evaluations_after_filtering"
        ]
        self.total_NO_MC_steps = additional["total_NO_MC_steps"]

        self.previous_new_points = additional["previous_new_points"]

        self.current_iteration = additional["current_iteration"]
        self.current_epoch_in_iteration = additional["current_epoch_in_iteration"]
        self.current_epoch = additional["current_epoch"]
        self.current_progress_bar = tqdm(
            total=EnsembleALManager.get_iteration_length(self.current_iteration),
            desc="Training by probability"
            if self.current_iteration != 0
            else "Training by example",
            mininterval=30,
        )

        return (
            additional["new_points_training"],
            additional["new_points_test"],
            additional["new_points_training_group_indices"],
            additional["new_points_test_group_indices"],
        )

    def train_ensemble_find_high_error_points_create_checkpoint(
        self,
        add_high_error_points: bool = True,
        silence_stopped_print: bool = False,
        reset_free_energy_models: bool = True,
        analysis_output_dir: str = None,
        save_checkpoint_dir: str = None,
        print_instead_of_logging: bool = False,
        plot_free_energy_training_curve_whole: bool = True,
        tag: str = "",
        starting_epoch: int = 0,
        plot_args: dict = {},
    ) -> bool:
        """First, trains the ensemble on the current AL dataset.
        Then, samples new high-error points from the ensemble.
        Then, creates a checkpoint.
        Finally, adds the new high-error points to the AL dataset.

        Args:
            add_high_error_points (bool, optional): Whether to find and add new high-error points to the AL dataset.
            silence_stopped_print (bool, optional): Whether to silence the print when the training is stopped.
            reset_free_energy_models (bool, optional): Whether to reset the free energy models of the ensemble
                before training. Defaults to True.
            analysis_output_dir (str, optional): The directory to save the analysis plots to. If None, the plots will be logged to wandb.
                Defaults to None.
            save_checkpoint_dir (str, optional): The directory to save the checkpoint to. If None, the checkpoint will be saved to the wandb run directory.
                Defaults to None.
            print_instead_of_logging (bool, optional): Whether to print metrics instead of logging to wandb. Defaults to False.
            plot_free_energy_training_curve_whole (bool, optional): Whether to plot the free energy training curve as a whole or log to wandb.
                Defaults to True.
            tag (str, optional): A tag to be added to the analysis plots. Defaults to "".
            starting_epoch (int, optional): The epoch to start training at. Defaults to 0.
            plot_args (dict, optional): Additional arguments to be passed to the plotting function. Defaults to {}.

        Returns:
            bool: True if the training should be stopped, False otherwise.
        """

        logging.info(f"Training free energy ensemble at epoch {self.current_epoch}...")

        start_time = time.time()

        if wandb.config.flow_architecture["filtering_type"] == "mirror":
            NO_evaluations = self.AL_dataset.get_current_evaluation_counters()[:, 0]
            current_xs_int = self.AL_dataset.get_current_xs_int()
            current_xs_int = current_xs_int[NO_evaluations > 0]
            L_chirality_mask = alternative_filter_chirality(
                current_xs_int,
                self.system.trafo_cpu,
                use_hydrogen_carbon_vector=True,
            )
        else:
            L_chirality_mask = None

        self.AL_ensemble.train_free_energy_matching(
            AL_dataset=self.AL_dataset,
            NO_epochs=wandb.config.NO_epochs_free_energy_training,
            batch_size=wandb.config.free_energy_ensemble["batch_size"],
            current_epoch=self.current_epoch,
            calculate_test_loss=True,
            target_log_prob_function=self.system.target_log_prob,
            plot_output_dir=analysis_output_dir,
            plot_training_curve_as_whole=plot_free_energy_training_curve_whole,
            starting_epoch=starting_epoch,
            reset_free_energy_models=reset_free_energy_models,
            L_chirality_mask=L_chirality_mask,
            invert_chirality_function=self.system.invert_cg_configurations
            if wandb.config.flow_architecture["filtering_type"] == "mirror"
            else None,
        )

        logging.info(
            f"Training free energy ensemble took {time.time()-start_time:.2f}s"
        )

        if add_high_error_points:
            # Sample new high-error points from the ensemble

            NO_starting_points_from_beginning = (
                int(
                    wandb.config.fraction_trajectories_start_at_beginning
                    * wandb.config.NO_new_points_per_iteration
                )
                if self.previous_new_points is not None
                else wandb.config.NO_new_points_per_iteration
            )
            NO_starting_points_previous = (
                (
                    wandb.config.NO_new_points_per_iteration
                    - NO_starting_points_from_beginning
                )
                if self.previous_new_points is not None
                else 0
            )

            if wandb.config.target_system["name"] == "mb":
                high_error_points_beginning = torch.full(
                    (NO_starting_points_from_beginning, 1),
                    -2.0,
                    device="cuda",  # Starting position in the very beginning: -2.0
                )
            else:
                # Start in the lowest minimum (upper left corner)
                high_error_points_beginning = (
                    torch.Tensor(
                        [-150.0 / 360 * 2 * math.pi, 155.0 / 360 * 2 * math.pi]
                    )
                    .cuda()[None, :]
                    .repeat((NO_starting_points_from_beginning, 1))
                )
                high_error_points_beginning *= -1.0  # different sign convention
                # Convert to scaled coordinates:
                high_error_points_beginning = self.system.convert_to_scaled_internal(
                    high_error_points_beginning, indices=wandb.config.CG_indices
                )

            if self.previous_new_points is None:
                high_error_points = high_error_points_beginning
            else:
                high_error_points_previous = self.previous_new_points[
                    torch.randperm(self.previous_new_points.shape[0])
                ][:NO_starting_points_previous].cuda()
                high_error_points = torch.cat(
                    [
                        high_error_points_beginning,
                        high_error_points_previous,
                    ],
                    dim=0,
                )

            assert (
                high_error_points.shape[0] == wandb.config.NO_new_points_per_iteration
            )

            logging.info("Sampling new high-error points...")
            start_time = time.time()

            starting_points = (
                high_error_points.detach().cpu().clone()
            )  # keep for plotting in run_analysis
            (
                high_error_points,
                NO_steps_this_iteration,
                NO_loop_iterations,
            ) = self.AL_sampler.sample_new_higherror_points(
                starting_points=high_error_points,
                NO_trajectories=wandb.config.NO_MC_trajectories,
                NO_points_to_find=wandb.config.NO_new_points_per_iteration
                * wandb.config.MC_diversity_multiplier,
                debugging=False,
            )

            end_time = time.time()

            if high_error_points is not None:  # time limit not reached
                force_stop = False

                if wandb.config.MC_diversity_multiplier != 1:
                    high_error_points = select_diverse_points(
                        high_error_points, wandb.config.NO_new_points_per_iteration
                    )

                logging.info(
                    f"Finished sampling {high_error_points.shape[0]} new points in {end_time-start_time:.2f}s taking {NO_steps_this_iteration} steps in {NO_loop_iterations} loop iterations"
                )

                wandb.log(
                    {
                        "NO_MC_steps_per_iteration": NO_steps_this_iteration,
                    },
                    step=self.current_epoch,
                )
                self.total_NO_MC_steps += NO_steps_this_iteration
                wandb.log(
                    {
                        "total_NO_MC_steps": self.total_NO_MC_steps,
                    },
                    step=self.current_epoch,
                )
                new_points = high_error_points.detach().cpu()
                self.previous_new_points = new_points

            else:
                force_stop = True
                new_points = None

        else:
            force_stop = False
            new_points = None
            starting_points = None

        if new_points is not None:
            (
                (original_train, original_test),
                (broadened_train, broadened_test),
                (broadened_group_indices_train, broadened_group_indices_test),
            ) = self.AL_dataset.spread_points_split_multiple_z(
                new_points, do_spread_points=True
            )

            if wandb.config.target_system["name"] == "aldp":  # wrapping
                original_train, original_test, broadened_train, broadened_test = (
                    self.system.wrap_cg_coordinates(original_train),
                    self.system.wrap_cg_coordinates(original_test),
                    self.system.wrap_cg_coordinates(broadened_train),
                    self.system.wrap_cg_coordinates(broadened_test),
                )

        stopped = self.create_checkpoint(
            new_points_training=broadened_train if new_points is not None else None,
            new_points_training_group_indices=broadened_group_indices_train
            if new_points is not None
            else None,
            new_points_test=broadened_test if new_points is not None else None,
            new_points_test_group_indices=broadened_group_indices_test
            if new_points is not None
            else None,
            new_points_training_centers=original_train
            if new_points is not None
            else None,
            MC_starting_points=starting_points,
            silence_stopped_print=silence_stopped_print,
            calculate_metrics=True,
            analysis_output_dir=analysis_output_dir,
            save_checkpoint_dir=save_checkpoint_dir,
            print_instead_of_logging=print_instead_of_logging,
            tag=tag,
            force_stop=force_stop,
            plot_args=plot_args,
        )

        if not stopped and new_points is not None:
            if wandb.config.flow_architecture["filtering_type"] == "mirror":
                broadened_train = torch.cat(
                    [
                        broadened_train,
                        self.system.invert_cg_configurations(broadened_train),
                    ],
                    dim=0,
                )
                broadened_group_indices_train = torch.cat(
                    [
                        broadened_group_indices_train,
                        broadened_group_indices_train
                        + torch.max(broadened_group_indices_train)
                        + 1,
                    ],
                    dim=0,
                )
                broadened_test = torch.cat(
                    [
                        broadened_test,
                        self.system.invert_cg_configurations(broadened_test),
                    ],
                    dim=0,
                )
                broadened_group_indices_test = torch.cat(
                    [
                        broadened_group_indices_test,
                        broadened_group_indices_test
                        + torch.max(broadened_group_indices_test)
                        + 1,
                    ],
                    dim=0,
                )

            self.AL_dataset.add_high_error_points(
                broadened_train,
                broadened_group_indices_train,
                put_on_train=True,
                iteration=self.current_iteration - 1,
            )
            self.AL_dataset.add_high_error_points(
                broadened_test,
                broadened_group_indices_test,
                put_on_train=False,
                iteration=self.current_iteration - 1,
            )

        return stopped

    def train_epoch(self) -> bool:
        """Trains one epoch.

        Returns:
            bool: True if the training should be stopped (stopping criterion satisfied), False otherwise.
        """

        if (
            self.current_iteration == 0
            and self.current_epoch_in_iteration
            < EnsembleALManager.get_iteration_length(0)
        ):
            self._train_epoch_by_example()

        else:
            if (
                self.current_iteration == 0
                and self.current_epoch_in_iteration
                == EnsembleALManager.get_iteration_length(0)
            ):  # Directly after training by example
                if not wandb.config.use_grid_conditioning:
                    # Add the starting points to the AL dataset
                    coords = self.starting_dataloader.tensors[0]

                    # Transform to CG space:
                    s = self.system.cartesian_to_internal(coords)[
                        :, self.system.CG_mask
                    ]

                else:
                    N_gridpoints = wandb.config.grid_conditioning_N_gridpoints
                    uniform_ranges = (
                        self.system.get_uniform_CG_ranges()
                    )  # [(min, max), ...] => Shape (D,2)
                    s = generate_grid(uniform_ranges, N_gridpoints)

                (
                    (original_train, original_test),
                    (broadened_train, broadened_test),
                    (broadened_group_indices_train, broadened_group_indices_test),
                ) = self.AL_dataset.spread_points_split_multiple_z(
                    s, do_spread_points=False
                )
                self.AL_dataset.add_high_error_points(
                    broadened_train,
                    broadened_group_indices_train,
                    put_on_train=True,
                    iteration=0,
                )
                self.AL_dataset.add_high_error_points(
                    broadened_test,
                    broadened_group_indices_test,
                    put_on_train=False,
                    iteration=0,
                )

                # Change the lr to the lr for training by probability
                for param_group in self.cond_flow_optimizer.param_groups:
                    param_group["lr"] = wandb.config.lr_probability
                self.initialize_lr_scheduler()

                self.current_iteration += 1
                self.current_epoch_in_iteration = 0
                self.current_progress_bar = tqdm(
                    total=EnsembleALManager.get_iteration_length(
                        self.current_iteration
                    ),
                    desc="Training by probability",
                    mininterval=30,
                )

                self.lr_scheduler.step()  # Make sure that in the beginning of the first iteration, the lr is also changed

                # yappi.set_clock_type("wall")
                # yappi.start()

                if (
                    self.current_epoch != 0
                ):  # Only create a checkpoint if we did actually train at least one epoch by example
                    # Create a checkpoint after training by example
                    self.create_checkpoint(
                        new_points_training=self.AL_dataset.get_current_s(),  # only the training points, not the test ones (for the analysis plots)
                        new_points_test=self.AL_dataset.get_current_s_test(),
                        silence_stopped_print=True,
                        calculate_metrics=False,
                        s_samples=s,  # Provide them, do not sample here!
                        plot_predicted_F=False,
                        display_training_points=False,
                        n_bins_backmapping=10,  # not many points yet
                    )

            elif (
                self.current_epoch_in_iteration
                == EnsembleALManager.get_iteration_length(self.current_iteration)
                or (
                    wandb.config.use_grid_conditioning
                    and (
                        self.current_epoch_in_iteration
                        % wandb.config.grid_conditioning_checkpoint_freq
                    )
                    == 0
                )
            ):  # After each iteration by probability
                # After each iteration, each point should have been evaluated at least once.
                # Otherwise, there is some kind of problem somewhere.
                # assert self.AL_dataset.get_min_NO_evaluations() > 0 # Disable for now

                if wandb.config.use_grid_conditioning:
                    assert self.current_iteration == 1

                    self.train_ensemble_find_high_error_points_create_checkpoint(
                        add_high_error_points=False,
                    )

                    if (
                        self.current_epoch_in_iteration
                        == EnsembleALManager.get_iteration_length(
                            self.current_iteration
                        )
                    ):  # stop
                        return True

                else:
                    self.current_iteration += 1
                    self.current_epoch_in_iteration = 0
                    self.current_progress_bar = tqdm(
                        total=EnsembleALManager.get_iteration_length(
                            self.current_iteration
                        ),
                        desc="Training by probability",
                        mininterval=30,
                    )

                    stopped = (
                        self.train_ensemble_find_high_error_points_create_checkpoint(
                            add_high_error_points=True,
                        )
                    )

                    if stopped:  # Threshold reached, stop training
                        return True

            if self.current_epoch_in_iteration < (
                wandb.config.training_by_probability_gradient_clipping[
                    "beginning_N_epochs"
                ]
                if self.current_iteration > 0
                else wandb.config.training_by_probability_gradient_clipping[
                    "beginning_N_epochs_first_iteration"
                ]
            ):
                clipping_value = wandb.config.training_by_probability_gradient_clipping[
                    "clipping_value_beginning_of_iteration"
                ]
            else:
                clipping_value = wandb.config.training_by_probability_gradient_clipping[
                    "clipping_value"
                ]

            # start = time.time()
            (
                NO_evaluations,
                NO_evaluations_after_filtering,
            ) = train_epoch_reweighted_by_probability(
                cond_flow=self.cond_flow,
                optimizer=self.cond_flow_optimizer,
                epoch=self.current_epoch,
                AL_dataset=self.AL_dataset,
                system=self.system,
                gradient_clipping_value=clipping_value,
            )
            # logging.info(
            #    f"Training epoch {self.current_epoch} took {time.time()-start:.2f}s"
            # )

            # start = time.time()
            test_loss = calculate_test_KL_by_probability_loss(
                cond_flow=self.cond_flow,
                AL_dataset=self.AL_dataset,
                system=self.system,
            )
            # logging.info(
            #    f"Calculating test loss of epoch {self.current_epoch} took {time.time()-start:.2f}s"
            # )

            self.total_NO_evaluations += NO_evaluations
            self.total_NO_evaluations_after_filtering += NO_evaluations_after_filtering

            wandb.log(
                {
                    "test_KL_by_probability_loss": test_loss,
                    "total_NO_evaluations": self.total_NO_evaluations,
                    "total_NO_evaluations_after_filtering": self.total_NO_evaluations_after_filtering,
                },
                step=self.current_epoch,
            )

        self.current_epoch_in_iteration += 1
        self.current_epoch += 1
        self.current_progress_bar.update(1)

        if (
            wandb.config.get("warmup_epochs_by_probability") is not None
            and self.lr_scheduler is not None
        ):
            self.lr_scheduler.step()  # Update lr

        # if self.current_iteration == 1 and self.current_epoch_in_iteration == 10:
        #    yappi.stop()
        #    yappi.get_func_stats().print_all()
        #    exit()

        return False
