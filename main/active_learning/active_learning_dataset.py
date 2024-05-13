import torch
import wandb
from typing import Tuple
from main.models.flow_base import ConditionalFlowBase
import logging
import math


class ActiveLearningDataset:
    def __init__(
        self,
        cond_flow: ConditionalFlowBase,
        starting_size: int = 5000,
        batch_size: int = 125,
        test_dataset_fraction: float = 0.1,
    ):
        """Dataset for active learning.

        Args:
            cond_flow (ConditionalFlowBase): Conditional normalizing flow.
            starting_size (int, optional): Starting size of the dataset. This can dynamically grow. Defaults to 5000.
            batch_size (int, optional): Batch size. Defaults to 1024.
            test_dataset_fraction (float, optional): Fraction of the dataset to use for testing. Defaults to 0.1.
        """

        ndim_z = cond_flow.q0.ndim
        ndim_s = len(wandb.config.CG_indices)
        ndim_x_int = ndim_z + ndim_s

        training_NO_samples = int(starting_size * (1 - test_dataset_fraction))

        self.xs_int = torch.empty((training_NO_samples, ndim_x_int))
        self.target_log_prob = torch.empty((training_NO_samples, 1))
        self.grads = torch.empty((training_NO_samples, ndim_x_int))
        self.evaluation_counters = torch.zeros(
            (training_NO_samples, 1), dtype=torch.int64
        )  # Indicates how many times the x and E values have already been newly assigned to this datapoint

        self.log_pxgiveny_0 = torch.empty(
            (training_NO_samples, 1)
        )  # needed for force-matching

        self.s = torch.empty((training_NO_samples, ndim_s))
        self.added_in_iteration = torch.empty((training_NO_samples,), dtype=torch.int64)
        self.z0s = torch.empty((training_NO_samples, ndim_z))
        self.s_group_indices = torch.empty((training_NO_samples,), dtype=torch.int64)

        self.s_test = torch.empty((starting_size - training_NO_samples, ndim_s))
        self.test_added_in_iteration = torch.empty(
            (starting_size - training_NO_samples,), dtype=torch.int64
        )
        self.s_test_group_indices = torch.empty(
            (self.s_test.shape[0],), dtype=torch.int64
        )

        self.current_NO_samples = 0
        self.current_NO_samples_test = 0

        self.cond_flow = cond_flow
        self.batch_size = batch_size
        self.test_dataset_fraction = test_dataset_fraction

    def get_current_xs_int(self) -> torch.Tensor:
        return self.xs_int[: self.current_NO_samples]

    def get_current_s(self) -> torch.Tensor:
        return self.s[: self.current_NO_samples]

    def get_current_added_in_iteration(self) -> torch.Tensor:
        return self.added_in_iteration[: self.current_NO_samples]

    def get_current_s_group_indices(self) -> torch.Tensor:
        return self.s_group_indices[: self.current_NO_samples]

    def get_current_z0s(self) -> torch.Tensor:
        return self.z0s[: self.current_NO_samples]

    def get_current_s_test(self) -> torch.Tensor:
        return self.s_test[: self.current_NO_samples_test]

    def get_current_test_added_in_iteration(self) -> torch.Tensor:
        return self.test_added_in_iteration[: self.current_NO_samples_test]

    def get_current_s_test_group_indices(self) -> torch.Tensor:
        return self.s_test_group_indices[: self.current_NO_samples_test]

    def get_current_target_log_prob(self) -> torch.Tensor:
        return self.target_log_prob[: self.current_NO_samples]

    def get_current_grads(self) -> torch.Tensor:
        return self.grads[: self.current_NO_samples]

    def get_current_evaluation_counters(self) -> torch.Tensor:
        return self.evaluation_counters[: self.current_NO_samples]

    def get_current_log_pxgiveny_0(
        self,
    ) -> torch.Tensor:
        return self.log_pxgiveny_0[: self.current_NO_samples]

    def _spread_points(self, starting_points: torch.Tensor) -> torch.Tensor:
        """Spread the points in `starting_points` using the configured distribution.

        Args:
            starting_points (torch.Tensor): The starting points, shape (N,D)

        Returns:
            torch.Tensor: Broadened points, shape (N*multiplier,D)
        """

        multiplier = wandb.config.new_point_spreading["multiplier"]
        spreading_type = wandb.config.new_point_spreading["type"]

        if spreading_type is None:
            return starting_points

        starting_points_expanded = starting_points.unsqueeze(1).expand(
            -1, multiplier, -1  # N, multiplier, D
        )
        mapping_array = (
            torch.arange(starting_points.size(0))
            .unsqueeze(1)
            .repeat(1, multiplier)
            .flatten()
        )

        if spreading_type == "gaussian":
            samples = torch.normal(
                mean=starting_points_expanded,
                std=wandb.config.new_point_spreading["scale"],
            )

        elif spreading_type == "uniform":
            radius = wandb.config.new_point_spreading["scale"]

            if starting_points_expanded.shape[-1] == 1:
                samples = (
                    torch.rand(starting_points_expanded.shape) * 2 - 1
                ) * radius + starting_points_expanded
            elif starting_points_expanded.shape[-1] == 2:
                angles = (
                    torch.rand(starting_points_expanded.shape[:-1]) * 2 * math.pi
                )  # Random angles
                r = (
                    torch.sqrt(torch.rand(starting_points_expanded.shape[:-1])) * radius
                )  # Random radii

                # Convert polar coordinates to Cartesian coordinates
                offset_x = r * torch.cos(angles)
                offset_y = r * torch.sin(angles)

                # Offset the starting points
                samples = torch.stack(
                    [
                        starting_points_expanded[..., 0] + offset_x,
                        starting_points_expanded[..., 1] + offset_y,
                    ],
                    dim=-1,
                )

            else:
                raise ValueError("Uniform spreading is only implemented for 1D and 2D.")

        else:
            raise ValueError(f"Unknown spreading type {spreading_type}")

        sampled_points = samples.reshape(-1, starting_points.shape[1])

        return sampled_points, mapping_array

    def spread_points_split_multiple_z(
        self, s: torch.Tensor, do_spread_points: bool = True
    ):
        if do_spread_points:
            s_spread, mapping_array = self._spread_points(s)
        else:
            s_spread = s
            mapping_array = torch.arange(s.shape[0])

        # Shuffle s
        permutation = torch.randperm(s_spread.shape[0])
        s_spread = s_spread[permutation]
        mapping_array = mapping_array[permutation]

        N = s_spread.shape[0]
        N_training = int(N * (1 - self.test_dataset_fraction))

        spread_datasets = [s_spread[:N_training], s_spread[N_training:]]
        broadened_datasets_multiple_z = []
        broadened_datasets_multiple_z_group_indices = []

        for dataset in spread_datasets:
            broadened_datasets_multiple_z_group_indices.append(
                torch.arange(dataset.shape[0]).repeat((wandb.config.number_of_z_per_y,))
            )
            dataset = dataset.repeat((wandb.config.number_of_z_per_y, 1))
            broadened_datasets_multiple_z.append(dataset)

        return (
            [
                s[mapping_array[:N_training].unique()],
                s[mapping_array[N_training:].unique()],
            ],
            broadened_datasets_multiple_z,
            broadened_datasets_multiple_z_group_indices,
        )

    def add_high_error_points(
        self,
        s: torch.Tensor,
        s_group_indices: torch.Tensor,
        iteration: int,
        put_on_train: bool = True,
    ):
        """Add new points to the dataset.

        Args:
            s (torch.Tensor): s values of the new points.
            s_group_indices (torch.Tensor): Group indices of the new points, indicating which entries have the same s value.
            iteration (int): Current iteration of the active learning algorithm after which the new points are being added.
            put_on_train (bool, optional): Whether to put the points on the training set or the test set. Defaults to True.
        """

        current_group_indices = (
            self.get_current_s_group_indices()
            if put_on_train
            else self.get_current_s_test_group_indices()
        )
        current_NO_samples = (
            self.current_NO_samples if put_on_train else self.current_NO_samples_test
        )

        while True:
            if current_NO_samples + s.shape[0] > (
                self.s.shape[0] if put_on_train else self.s_test.shape[0]
            ):
                self.grow_dataset()
            else:
                break

        dataset = self.s if put_on_train else self.s_test
        dataset_group_indices = (
            self.s_group_indices if put_on_train else self.s_test_group_indices
        )
        iteration_tensor = (
            self.added_in_iteration if put_on_train else self.test_added_in_iteration
        )

        dataset[current_NO_samples : current_NO_samples + s.shape[0]] = s
        iteration_tensor[
            current_NO_samples : current_NO_samples + s.shape[0]
        ] = iteration

        next_group_index = (
            (torch.max(current_group_indices) + 1) if current_NO_samples > 0 else 0
        )
        dataset_group_indices[current_NO_samples : current_NO_samples + s.shape[0]] = (
            s_group_indices - torch.min(s_group_indices)
        ) + next_group_index

        if put_on_train:
            self.current_NO_samples += s.shape[0]
        else:
            self.current_NO_samples_test += s.shape[0]

        logging.info(
            f"\nAdded {s.shape[0]} points to {'training' if put_on_train else 'test'} dataset"
        )

    def shuffle(self):
        """Shuffle the dataset."""

        # Train set
        indices_before = torch.arange(self.current_NO_samples)

        if not wandb.config.AL_dataset_grow_settings[
            "yield_mainly_samples_from_latest_iteration"
        ]:
            indices_new = torch.randperm(self.current_NO_samples)

        else:
            fraction_old_data = wandb.config.AL_dataset_grow_settings[
                "fraction_added_from_previous_iterations"
            ]

            added_in_iteration = self.get_current_added_in_iteration()

            # Identify the samples from the latest AL iteration
            latest_iteration = torch.max(added_in_iteration)
            latest_samples_indices = torch.where(
                added_in_iteration == latest_iteration
            )[0]

            # Calculate 20% of the size of the latest iteration
            num_additional_samples = int(
                fraction_old_data * latest_samples_indices.size(0)
            )

            # Indices of samples from previous iterations
            previous_samples_indices = torch.where(
                added_in_iteration != latest_iteration
            )[0]

            # Randomly select the required number of indices from previous iterations
            if num_additional_samples > 0 and previous_samples_indices.size(0) > 0:
                additional_indices = previous_samples_indices[
                    torch.randperm(previous_samples_indices.size(0))[
                        :num_additional_samples
                    ]
                ]
            else:
                additional_indices = torch.tensor([], dtype=torch.int64)

            # Create the first part of the index tensor
            first_part_indices = torch.cat([latest_samples_indices, additional_indices])

            # Shuffle this first part
            shuffled_first_part_indices = first_part_indices[
                torch.randperm(first_part_indices.size(0))
            ]

            all_indices = torch.arange(self.current_NO_samples, dtype=torch.int64)
            mask = ~torch.isin(all_indices, shuffled_first_part_indices)
            remaining_indices = all_indices[mask]

            # Shuffle the remaining indices
            shuffled_remaining_indices = remaining_indices[
                torch.randperm(remaining_indices.size(0))
            ]

            # Concatenate to form the final shuffle index tensor
            indices_new = torch.cat(
                [shuffled_first_part_indices, shuffled_remaining_indices]
            )

        self.xs_int[indices_before] = self.xs_int[indices_new]
        self.target_log_prob[indices_before] = self.target_log_prob[indices_new]
        self.grads[indices_before] = self.grads[indices_new]
        self.evaluation_counters[indices_before] = self.evaluation_counters[indices_new]
        self.log_pxgiveny_0[indices_before] = self.log_pxgiveny_0[indices_new]
        self.s[indices_before] = self.s[indices_new]
        self.added_in_iteration[indices_before] = self.added_in_iteration[indices_new]
        self.s_group_indices[indices_before] = self.s_group_indices[indices_new]
        self.z0s[indices_before] = self.z0s[indices_new]

        # Test set
        indices_new_test = torch.randperm(self.current_NO_samples_test)
        indices_before_test = torch.arange(self.current_NO_samples_test)
        self.s_test[indices_before_test] = self.s_test[indices_new_test]
        self.test_added_in_iteration[
            indices_before_test
        ] = self.test_added_in_iteration[indices_new_test]
        self.s_test_group_indices[indices_before_test] = self.s_test_group_indices[
            indices_new_test
        ]

    def grow_dataset(self, N_train: int = None, N_test: int = None):
        """Grow the tensors of training and test dataset.

        Args:
            N_train (int, optional): Number of new training points.
                If None, the tensor sizes will be doubled. Defaults to None.
            N_test (int, optional): Number of new test points.
                If None, the tensor sizes will be doubled. Defaults to None.
        """

        if N_train is None:
            N_train = self.xs_int.shape[0]

        if N_test is None:
            N_test = self.s_test.shape[0]

        self.xs_int = torch.cat(
            [
                self.xs_int,
                torch.empty((N_train, self.xs_int.shape[1]), dtype=self.xs_int.dtype),
            ],
            dim=0,
        )
        self.target_log_prob = torch.cat(
            [
                self.target_log_prob,
                torch.empty(
                    (N_train, self.target_log_prob.shape[1]),
                    dtype=self.target_log_prob.dtype,
                ),
            ],
            dim=0,
        )
        self.grads = torch.cat(
            [
                self.grads,
                torch.empty((N_train, self.grads.shape[1]), dtype=self.grads.dtype),
            ],
            dim=0,
        )
        self.evaluation_counters = torch.cat(
            [
                self.evaluation_counters,
                torch.zeros(
                    (N_train, self.evaluation_counters.shape[1]),
                    dtype=self.evaluation_counters.dtype,
                ),
            ],
            dim=0,
        )
        self.log_pxgiveny_0 = torch.cat(
            [
                self.log_pxgiveny_0,
                torch.empty(
                    (N_train, self.log_pxgiveny_0.shape[1]),
                    dtype=self.log_pxgiveny_0.dtype,
                ),
            ],
            dim=0,
        )
        self.s = torch.cat(
            [
                self.s,
                torch.empty((N_train, self.s.shape[1]), dtype=self.s.dtype),
            ],
            dim=0,
        )
        self.added_in_iteration = torch.cat(
            [
                self.added_in_iteration,
                torch.empty((N_train,), dtype=self.added_in_iteration.dtype),
            ],
            dim=0,
        )
        self.s_group_indices = torch.cat(
            [
                self.s_group_indices,
                torch.empty((N_train,), dtype=self.s_group_indices.dtype),
            ],
            dim=0,
        )
        self.z0s = torch.cat(
            [
                self.z0s,
                torch.empty((N_train, self.z0s.shape[1]), dtype=self.z0s.dtype),
            ],
            dim=0,
        )

        self.s_test = torch.cat(
            [
                self.s_test,
                torch.empty((N_test, self.s_test.shape[1]), dtype=self.s_test.dtype),
            ],
            dim=0,
        )
        self.test_added_in_iteration = torch.cat(
            [
                self.test_added_in_iteration,
                torch.empty((N_test,), dtype=self.test_added_in_iteration.dtype),
            ],
            dim=0,
        )
        self.s_test_group_indices = torch.cat(
            [
                self.s_test_group_indices,
                torch.empty((N_test,), dtype=self.s_test_group_indices.dtype),
            ],
            dim=0,
        )

    def save_state(self, path: str):
        """Save the current state of the dataset to a file. This does include the INN."""

        data = {
            "xs_int": self.xs_int[: self.current_NO_samples],
            "target_log_prob": self.target_log_prob[: self.current_NO_samples],
            "grads": self.grads[: self.current_NO_samples],
            "evaluation_counters": self.evaluation_counters[: self.current_NO_samples],
            "log_pxgiveny_0": self.log_pxgiveny_0[: self.current_NO_samples],
            "s": self.s[: self.current_NO_samples],
            "added_in_iteration": self.added_in_iteration[: self.current_NO_samples],
            "s_group_indices": self.s_group_indices[: self.current_NO_samples],
            "z0s": self.z0s[: self.current_NO_samples],
            "s_test": self.s_test[: self.current_NO_samples_test],
            "test_added_in_iteration": self.test_added_in_iteration[
                : self.current_NO_samples_test
            ],
            "s_test_group_indices": self.s_test_group_indices[
                : self.current_NO_samples_test
            ],
            "current_NO_samples": self.current_NO_samples,
            "current_NO_samples_test": self.current_NO_samples_test,
            "batch_size": self.batch_size,
            "inn_state": self.cond_flow.state_dict(),
        }

        torch.save(data, path)

    def load_state(self, path: str):
        """Load the dataset from a file. This does not include the invertible model."""

        data = torch.load(path)

        self.xs_int = data["xs_int"]
        self.s = data["s"]
        self.added_in_iteration = data["added_in_iteration"]
        self.s_group_indices = data["s_group_indices"]
        self.s_test = data["s_test"]
        self.test_added_in_iteration = data["test_added_in_iteration"]
        self.s_test_group_indices = data["s_test_group_indices"]
        self.target_log_prob = data["target_log_prob"]
        self.grads = data["grads"]
        self.evaluation_counters = data["evaluation_counters"]
        self.log_pxgiveny_0 = data["log_pxgiveny_0"]
        self.z0s = data["z0s"]
        self.current_NO_samples = data["current_NO_samples"]
        self.current_NO_samples_test = data["current_NO_samples_test"]
        self.batch_size = data["batch_size"]
        self.cond_flow.load_state_dict(data["inn_state"])

    def __len__(self):
        """Returns the number of batches in the dataset."""

        if not wandb.config.AL_dataset_grow_settings[
            "yield_mainly_samples_from_latest_iteration"
        ]:
            return self.current_NO_samples // self.batch_size
        else:
            # Number of samples from the latest iteration:
            latest_iteration_index = torch.max(self.get_current_added_in_iteration())
            N_latest_iteration = torch.sum(
                self.get_current_added_in_iteration() == latest_iteration_index
            )

            available_N_previous_iterations = (
                self.current_NO_samples - N_latest_iteration
            )
            # Number of samples to take from the previous iterations:
            to_take_N_previous_iterations = (
                wandb.config.AL_dataset_grow_settings[
                    "fraction_added_from_previous_iterations"
                ]
                * N_latest_iteration
            )
            to_take_N_previous_iterations = min(
                to_take_N_previous_iterations, available_N_previous_iterations
            )

            return int(
                (N_latest_iteration + to_take_N_previous_iterations) // self.batch_size
            )

    def get_min_NO_evaluations(self):
        """Returns the minimum number of evaluations that have been performed on a datapoint."""
        return torch.min(self.get_current_evaluation_counters()).item()

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Returns the idx-th batch of the dataset.

        Args:
            idx (int): Index of the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                current_target_log_prob, current_grads, current_xs_int, current_s, current_z0s, current_log_pxgiveny_0, need_evaluation
        """

        if idx >= len(self):
            raise IndexError("Index out of bounds")

        current_target_log_prob = self.target_log_prob[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        current_grads = self.grads[idx * self.batch_size : (idx + 1) * self.batch_size]
        current_xs_int = self.xs_int[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        current_evaluation_counters = self.evaluation_counters[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        current_s = self.s[idx * self.batch_size : (idx + 1) * self.batch_size]
        current_z0s = self.z0s[idx * self.batch_size : (idx + 1) * self.batch_size]
        current_log_pxgiveny_0 = self.log_pxgiveny_0[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        may_be_evaluated = (
            current_evaluation_counters
            < wandb.config.total_number_of_evaluations_before_adding_to_buffer
        ).squeeze()

        definitely_need_evaluation = (current_evaluation_counters == 0).squeeze()

        p = wandb.config.resampling_probability
        random_vector = torch.rand(may_be_evaluated.shape)
        need_evaluation = (
            (random_vector < p) & may_be_evaluated
        ) | definitely_need_evaluation

        ##### If we (re-)evaluate it newly anyways, we can just also sample a new z_0:
        sampled = self.cond_flow.q0.sample(len(current_z0s[need_evaluation]))
        current_z0s[need_evaluation] = sampled.cpu()

        # However, for those that do not get newly evaluated, we are not allowed to change the z_0 (or p(x|y))

        current_evaluation_counters[need_evaluation] += 1

        return (
            current_target_log_prob,
            current_grads,
            current_xs_int,
            current_s,
            current_z0s,
            current_log_pxgiveny_0,
            need_evaluation,
            current_evaluation_counters,
        )
