import torch
from torch import optim
from typing import List
from main.models.free_energy_models import FCFreeEnergyModel
from main.utils.FastTensorDataLoader import FastTensorDataLoader
from main.utils.sampling import rejection_sampling
import matplotlib.pyplot as plt
import wandb
import main.utils.matplotlib_helpers
from main.active_learning.active_learning_dataset import ActiveLearningDataset
from typing import Tuple
from main.utils.FastTensorDataLoader import FastTensorDataLoader
from main.utils.newline_tqdm import NewlineTqdm as tqdm
import numpy as np
from main.utils.wandb_helpers import save_fig_with_dpi
import os
import logging
from functorch import combine_state_for_ensemble
from functorch import vmap
import time
import torchinfo
from main.utils.matplotlib_helpers import set_default_paras


class FreeEnergyEnsemble:
    def __init__(self, output_force: bool = True, learning_rate: float = 0.1):
        """Create an ensemble of free energy models.

        Args:
            output_force (bool, optional): Whether the output of the models is a tuple (energy, force) or just the energy. Defaults to True.
            learning_rate (float, optional): Learning rate. Defaults to 0.1.
        """

        self.output_force = output_force

        self.optimizer = None
        self.lr_scheduler = None
        self.lr = learning_rate

        self.combined_model = None
        self.combined_params = None
        self.combined_buffers = None

        self.models = []

        self.use_MC_dropout = wandb.config.free_energy_ensemble.get("MC_dropout", False)

    def combine_models(self, models):
        """Combine the models in the ensemble into a single stacked model for parallelization with vmap"""

        (
            self.combined_model,
            self.combined_params,
            self.combined_buffers,
        ) = combine_state_for_ensemble(models)

        for param in self.combined_params:
            param.requires_grad = True

        self.optimizer = optim.Adam(self.combined_params, lr=self.lr)
        # self.optimizer = optim.SGD(self.combined_params, lr=self.lr)

        lr_scheduling_info = wandb.config.get("free_energy_lr_scheduling", None)

        if lr_scheduling_info is not None:
            lr_scheduling_type = lr_scheduling_info["type"]

            if lr_scheduling_type == "step":
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=lr_scheduling_info["step_size"],
                    gamma=lr_scheduling_info["gamma"],
                )
            elif lr_scheduling_type == "cosine":
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=wandb.config.NO_epochs_free_energy_training,
                    eta_min=0,
                )
            else:
                raise NotImplementedError(
                    f"LR scheduling type {lr_scheduling_type} not implemented."
                )

    def add_FC_free_energy_models(
        self,
        NO_models: int,
        size_of_latent_space: int = 1,
        hidden_layers: List[int] = [128],
        device: str = "cuda",
        seeds: List[int] = None,
        use_2pi_periodic_representation: bool = False,
        standard_scaler: tuple = None,
    ):
        self.models = []

        for i in range(NO_models):
            seed = seeds[i] if seeds is not None else None
            model = FCFreeEnergyModel(
                size_of_latent_space=size_of_latent_space,
                hidden_layers=hidden_layers,
                device=device,
                seed=seed,
                output_force=self.output_force,
                use_2pi_periodic_representation=use_2pi_periodic_representation,
                standard_scaler=standard_scaler,
                dropout=wandb.config.free_energy_ensemble.get("dropout_rate", 0.0),
            )

            if i == 0:
                # Print number of parameters
                logging.info(
                    "\nNumber of parameters in free energy model: "
                    + str(sum([np.prod(p.size()) for p in model.parameters()]))
                    + "\n",
                )

            self.models.append(model)

        self.combine_models(self.models)

    def reset_FC_free_energy_models(
        self, seeds: List[int] = None, standard_scaler: tuple = None
    ):
        for i in reversed(range(len(self.models))):
            seed = seeds[i] if seeds is not None else None
            self.models[i].initialize_weights(seed=seed)  # reset weights

            if standard_scaler is not None:
                self.models[i].scaler_m = standard_scaler[0]
                self.models[i].scaler_s = standard_scaler[1]

        self.combine_models(self.models)

    def set_lr(self, lr: float):
        self.lr = lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def forward(
        self,
        x_int: torch.Tensor,
        same_batch_all_models: bool = True,
        select_force: bool = False,
        MC_samples: int = 1,
        same_seed_MC_samples: bool = False,
    ) -> torch.Tensor:
        """Perform a forward pass for each model in the ensemble.

        Args:
            x_int (torch.Tensor): Input tensor.
            same_batch_all_models (bool, optional): Whether to use the same minibatch for all models.
                Otherwise, x_int needs to have an additional dimension for the model index in front.
                Defaults to True.
            select_force (bool, optional): Whether to select the force from the output tuple.
                Otherwise, the energy is selected. Defaults to False.
            MC_samples (int, optional): Number of MC samples to use for MC dropout. Defaults to 1 for training.
            same_seed_MC_samples (bool, optional): Whether to use the same seed for all MC samples.
                Defaults to False.

        Returns:
            torch.Tensor: Output tensor.
        """

        if select_force and not self.output_force:
            raise ValueError(
                "The ensemble was not created with output_force=True, so the force cannot be selected."
            )

        if self.use_MC_dropout and len(self.models) > 1:
            raise NotImplementedError("MC dropout is not compatible with ensembles.")

        if not self.use_MC_dropout:
            if same_batch_all_models:
                output = vmap(
                    self.combined_model, (0, 0, None), randomness="different"
                )(self.combined_params, self.combined_buffers, x_int)
            else:
                output = vmap(self.combined_model, randomness="different")(
                    self.combined_params, self.combined_buffers, x_int
                )

        else:
            # Perform MC dropout

            outputs = []

            previous_state = torch.cuda.get_rng_state()

            # Make sure that model is always in train mode
            self.combined_model.train()

            for i in range(MC_samples):
                if same_seed_MC_samples:
                    torch.manual_seed(123 + i)

                if same_batch_all_models:
                    new_output = vmap(
                        self.combined_model, (0, 0, None), randomness="different"
                    )(
                        self.combined_params,
                        self.combined_buffers,
                        x_int,
                    )
                    outputs.append(new_output[0, :])

                else:
                    new_output = vmap(self.combined_model, randomness="different")(
                        self.combined_params,
                        self.combined_buffers,
                        x_int,
                    )
                    outputs.append(new_output[0, :])

            output = torch.stack(outputs, dim=0)

            if same_seed_MC_samples:
                torch.cuda.set_rng_state(previous_state)

        if self.output_force:  # if output is a tuple of tensors
            output = output[0 if not select_force else 1]

        return output

    def std_dev(
        self,
        x_int: torch.Tensor,
        same_batch_all_models: bool = True,
        select_force: bool = False,
        MC_samples: int = 10,
        same_seed_MC_samples: bool = True,
    ) -> torch.Tensor:
        """Calculate the standard deviation of the ensemble's outputs.

        Args:
            x_int (torch.Tensor): Input tensor.
            same_batch_all_models (bool, optional): Whether to use the same minibatch for all models.
                Otherwise, x_int needs to have an additional dimension for the model index in front.
                Defaults to True.
            select_force (bool, optional): Whether to select the force from the output tuple.
                Otherwise, the energy is selected. Defaults to False.
            MC_samples (int, optional): Number of MC samples to use for MC dropout. Defaults to 10.
            same_seed_MC_samples (bool, optional): Whether to use the same seed for all MC samples.
                Defaults to True.

        Returns:
            torch.Tensor: Standard deviation of the ensemble's outputs.
        """

        outputs = self.forward(
            x_int,
            same_batch_all_models=same_batch_all_models,
            select_force=select_force,
            MC_samples=MC_samples,
            same_seed_MC_samples=same_seed_MC_samples,
        )

        # calculate and return the standard deviation of the ensemble's outputs
        std_dev = torch.std(outputs, dim=0)

        return std_dev

    def mean(
        self,
        x_int: torch.Tensor,
        same_batch_all_models: bool = True,
        select_force: bool = False,
        MC_samples: int = 10,
        same_seed_MC_samples: bool = True,
    ) -> torch.Tensor:
        """Calculate the mean of the ensemble's outputs.

        Args:
            x_int (torch.Tensor): Input tensor.
            same_batch_all_models (bool, optional): Whether to use the same minibatch for all models.
                Otherwise, x_int needs to have an additional dimension for the model index in front.
                Defaults to True.
            select_force (bool, optional): Whether to select the force from the output tuple.
                Otherwise, the energy is selected. Defaults to False.
            MC_samples (int, optional): Number of MC samples to use for MC dropout. Defaults to 10.
            same_seed_MC_samples (bool, optional): Whether to use the same seed for all MC samples.
                Defaults to True.

        Returns:
            torch.Tensor: Mean of the ensemble's outputs.
        """

        outputs = self.forward(
            x_int,
            same_batch_all_models=same_batch_all_models,
            select_force=select_force,
            MC_samples=MC_samples,
            same_seed_MC_samples=same_seed_MC_samples,
        )

        # calculate and return the mean of the ensemble's outputs
        mean = torch.mean(outputs, dim=0)

        return mean

    def mean_and_std_dev(
        self,
        x_int: torch.Tensor,
        same_batch_all_models: bool = True,
        select_force: bool = False,
        MC_samples: int = 10,
        same_seed_MC_samples: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the mean and standard deviation of the ensemble's outputs.

        Args:
            x_int (torch.Tensor): Input tensor.
            same_batch_all_models (bool, optional): Whether to use the same minibatch for all models.
                Otherwise, x_int needs to have an additional dimension for the model index in front.
                Defaults to True.
            select_force (bool, optional): Whether to select the force from the output tuple.
                Otherwise, the energy is selected. Defaults to False.
            MC_samples (int, optional): Number of MC samples to use for MC dropout. Defaults to 10.
            same_seed_MC_samples (bool, optional): Whether to use the same seed for all MC samples.
                Defaults to True.

        Returns:
            tuple: Tuple of mean and standard deviation.
        """

        outputs = self.forward(
            x_int,
            select_force=select_force,
            same_batch_all_models=same_batch_all_models,
            MC_samples=MC_samples,
            same_seed_MC_samples=same_seed_MC_samples,
        )

        # calculate and return the mean and standard deviation of the ensemble's outputs
        mean = torch.mean(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        return mean, std_dev

    def train_epoch(
        self,
        dataloader: FastTensorDataLoader,
        use_force_matching: bool = False,
        only_get_test_loss: bool = False,
    ) -> float:
        """Train the ensemble for one epoch.

        Args:
            dataloader: FastTensorDataLoader with tensors (inputs, targets).
                The input tensor and targets tensors should be of shape (batch_index, data_index, model_index).
            use_force_matching (bool, optional): Whether we are using force-matching (=> select the forces from the outputs of the models)
                Defaults to False.
            only_get_test_loss (bool, optional): Whether to only get the test loss and not train.
                Defaults to False.

        Returns:
            float: Mean loss over all batches.
        """

        with torch.set_grad_enabled(not only_get_test_loss):
            total_loss_per_model = np.zeros(len(self.models))
            counter = 0

            for inputs, targets in dataloader:
                inputs, targets = (
                    inputs.to("cuda"),
                    targets.to("cuda"),
                )

                # Move last dimension to the front => (model_index, batch_index, data_index)
                inputs = inputs.permute(2, 0, 1)
                targets = targets.permute(2, 0, 1)

                if not only_get_test_loss:
                    self.optimizer.zero_grad()

                outputs = self.forward(
                    inputs,
                    same_batch_all_models=False,
                )

                if not use_force_matching:
                    outputs = outputs[
                        :, :, None
                    ]  # add 1 if energy output (if force output, this is already dimension 2)

                if use_force_matching:
                    outputs_energy = outputs[0]  # Select the energy
                    outputs_force = outputs[1]  # Select the force
                else:
                    if isinstance(outputs, tuple):
                        outputs_energy = outputs[0]  # Select the energy
                    else:
                        outputs_energy = outputs

                if use_force_matching:
                    loss = torch.mean(
                        torch.sum((outputs_force - targets) ** 2, dim=2), dim=1
                    )
                    total_loss_per_model += (
                        loss.detach().cpu().numpy() * inputs.shape[0]
                    )

                    loss = torch.sum(loss, dim=0)
                else:
                    loss = torch.mean((outputs_energy - targets) ** 2, dim=[1, 2])
                    total_loss_per_model += (
                        loss.detach().cpu().numpy() * inputs.shape[0]
                    )

                    loss = torch.sum(loss, dim=0)

                counter += inputs.shape[0]

                # Check for nan or inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.info("Loss while training free energy model is nan or inf.")
                    continue

                if not only_get_test_loss:
                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

            if not only_get_test_loss and self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return total_loss_per_model / counter

    def train_force_matching(
        self,
        cg_configs: torch.Tensor,
        projected_forces: torch.Tensor,
        NO_epochs: int,
        batch_size: int = 125,
        current_epoch: int = None,
    ):
        """Train the model using force-matching.
        This does not use a AL_dataset, but rather cg_configs and projected forces directly.

        Args:
            cg_configs (torch.Tensor): CG configurations.
            projected_forces (torch.Tensor): Projected forces.
            NO_epochs (int): Number of epochs to train for.
            batch_size (int, optional): Batch size. Defaults to None.
            current_epoch (int, optional): Current epoch of the main outer cycle. If this is not None, a
                separate figure of the training curve is logged at the current outer epoch. If it is None,
                the loss is directly logged to wandb. Defaults to None.
        """

        raise NotImplementedError("Code needs to be updated!")

    def _prepare_dataloaders_free_energy_matching(
        self,
        cg_configs,
        target_log_prob,
        log_p_x_int_given_s,
        current_s_group_indices,
        batch_size: int,
        n_models: int,
        reset_free_energy_models: bool = True,
        L_chirality_mask: torch.Tensor = None,
        invert_chirality_function: callable = None,
    ):
        values_before_exp = -log_p_x_int_given_s + target_log_prob

        nan_selector = torch.isnan(values_before_exp)
        values_before_exp = values_before_exp[~nan_selector]

        values_before_exp -= torch.median(values_before_exp)  # avoid numerical issues
        values = torch.exp(values_before_exp)

        contracted_values = []
        cg_configs_contracted = []

        cg_configs = cg_configs[~nan_selector]

        if not wandb.config.flow_architecture["filtering_type"] == "mirror":
            contract_indices = current_s_group_indices[~nan_selector]
            contract_indices_unique = torch.unique(contract_indices)
        else:
            cg_configs[~L_chirality_mask] = invert_chirality_function(
                cg_configs[~L_chirality_mask]
            )
            _, contract_indices = torch.unique(
                torch.round(cg_configs, decimals=5),
                dim=0,
                return_inverse=True
                # cg_configs,
            )  # cannot use current_s_group_indices anymore due to mirroring of some of the configurations
            contract_indices_unique = torch.unique(contract_indices)

        for index in contract_indices_unique:
            selector = contract_indices == index

            current_values_to_contract = values[selector]

            if wandb.config.flow_architecture["filtering_type"] == "mirror":
                if current_values_to_contract.shape[0] < (
                    wandb.config.number_of_z_per_y / 2.0
                    - wandb.config.number_of_z_per_y * 0.2
                ):
                    continue

            k = wandb.config.free_energy_ensemble.get("clipping_k")
            if k is not None and k > 0:
                indices_to_clip = torch.sort(values[selector]).indices[-k:]
                current_values_to_contract[
                    indices_to_clip
                ] = current_values_to_contract[indices_to_clip][0]

            cg_configs_contracted.append(cg_configs[selector][0, :])

            integrated_value = torch.sum(current_values_to_contract)
            contracted_values.append(-torch.log(integrated_value))

        cg_configs = torch.stack(cg_configs_contracted, dim=0)

        if reset_free_energy_models:
            self.reset_FC_free_energy_models(
                seeds=None,
                standard_scaler=(
                    torch.mean(cg_configs, dim=0, keepdim=True).cuda()
                    if wandb.config.free_energy_ensemble["apply_mean_standard_scaler"]
                    else 0.0,
                    torch.std(cg_configs, dim=0, keepdim=True, unbiased=False).cuda()
                    if wandb.config.free_energy_ensemble["apply_std_standard_scaler"]
                    else 1.0,
                ),
            )

        targets = torch.FloatTensor(contracted_values)
        targets = targets[:, None]  # (N,1)

        ##### Prepare train-test splits #####

        train_fraction = 1 - wandb.config.free_energy_ensemble["test_dataset_fraction"]
        NO_train = int(train_fraction * cg_configs.shape[0])
        NO_test = cg_configs.shape[0] - NO_train

        train_cg_configs = torch.empty(NO_train, cg_configs.shape[1], n_models)
        train_targets = torch.empty(NO_train, targets.shape[1], n_models)

        test_cg_configs = torch.empty(NO_test, cg_configs.shape[1], n_models)
        test_targets = torch.empty(NO_test, targets.shape[1], n_models)

        if wandb.config.free_energy_ensemble["strategy"] == "bagging":
            # If bagging, use the same train-test split for all models
            indices = torch.randperm(cg_configs.shape[0])
            train_indices = indices[:NO_train]
            test_indices = indices[NO_train:]

        for i in range(n_models):
            if wandb.config.free_energy_ensemble["strategy"] == "fraction":
                indices = torch.randperm(cg_configs.shape[0])
                train_indices = indices[:NO_train]
                test_indices = indices[NO_train:]

                train_cg_configs[:, :, i] = cg_configs[train_indices]
                train_targets[:, :, i] = targets[train_indices]

                test_cg_configs[:, :, i] = cg_configs[test_indices]
                test_targets[:, :, i] = targets[test_indices]

            elif wandb.config.free_energy_ensemble["strategy"] == "bagging":
                # Use bootstrapping (sampling with replacement) for the train set

                # Randomly sample indices with replacement
                current_train_indices = torch.randint(
                    0,
                    train_indices.shape[0],
                    (train_indices.shape[0],),
                )

                train_cg_configs[:, :, i] = cg_configs[train_indices][
                    current_train_indices
                ]
                train_targets[:, :, i] = targets[train_indices][current_train_indices]

                test_cg_configs[:, :, i] = cg_configs[test_indices]
                test_targets[:, :, i] = targets[test_indices]

            else:
                raise NotImplementedError(
                    f"Strategy {wandb.config.free_energy_ensemble['strategy']} not implemented."
                )

        if False:
            # Plot the training data of the individual models
            fig = plt.figure()

            for i in range(n_models):
                plt.scatter(
                    train_cg_configs[:, 0, i].numpy(),
                    train_targets[:, 0, i].numpy(),
                    label=f"model {i}",
                    alpha=0.1,
                )
            plt.show()

        train_dataloader = FastTensorDataLoader(
            train_cg_configs,
            train_targets,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_dataloader = FastTensorDataLoader(
            test_cg_configs,
            test_targets,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_dataloader, test_dataloader

    def train_free_energy_matching(
        self,
        AL_dataset: ActiveLearningDataset,
        NO_epochs: int,
        batch_size: int = 125,
        current_epoch: int = None,
        calculate_test_loss: bool = False,
        target_log_prob_function: callable = None,
        plot_output_dir: str = None,
        plot_training_curve_as_whole: bool = False,
        starting_epoch: int = 0,
        reset_free_energy_models: bool = True,
        L_chirality_mask: torch.Tensor = None,
        invert_chirality_function: callable = None,
    ):
        """Train the model on an active learning dataset using the free energy matching approach.

        Args:
            AL_dataset (ActiveLearningDataset): Active learning dataset.
            NO_epochs (int): Number of epochs to train for.
            batch_size (int, optional): Batch size. Defaults to 125.
            current_epoch (int, optional): Current epoch of the main outer cycle.
            calculate_test_loss (bool, optional): Whether to additionally calculate the test loss in each epoch. Defaults to False.
            target_log_prob_function (callable, optional): Target log probability function from the System class. Defaults to None.
            plot_output_dir (str, optional): Directory to save training plots to. If None, plot pngs are logged to wandb.
                Defaults to None.
            plot_training_curve_as_whole (bool, optional): Whether to save the training curve as a whole as a png or using wandb.log
                for the individual points. Defaults to False.
            starting_epoch (int, optional): Epoch to start with. Defaults to 0.
            reset_free_energy_models (bool, optional): Whether to reset the weights of the free energy models before training.
                Defaults to True.
            L_chirality_mask (torch.Tensor, optional): Mask for the chirality of the L particles, only used for mirror chirality strategy. Defaults to None.
            invert_chirality_function (callable, optional): Function to invert the chirality of CG configurations, only used for mirror chirality strategy.
                Defaults to None.
        """

        # region Prepare training dataloader

        if target_log_prob_function is None and calculate_test_loss:
            raise ValueError(
                "If calculate_test_loss is True, target_log_prob_function must be given."
            )

        NO_evaluations = AL_dataset.get_current_evaluation_counters()[:, 0]

        cg_configs = AL_dataset.get_current_s()
        target_log_prob = AL_dataset.get_current_target_log_prob()[:, 0]
        log_pxgivens = AL_dataset.get_current_log_pxgiveny_0()[:, 0]
        current_s_group_indices = AL_dataset.get_current_s_group_indices()

        cg_configs = cg_configs[NO_evaluations > 0]
        target_log_prob = target_log_prob[NO_evaluations > 0]
        log_pxgivens = log_pxgivens[NO_evaluations > 0]
        current_s_group_indices = current_s_group_indices[NO_evaluations > 0]

        (
            dataloader_train,
            dataloader_test,
        ) = self._prepare_dataloaders_free_energy_matching(
            cg_configs=cg_configs,
            target_log_prob=target_log_prob,
            log_p_x_int_given_s=log_pxgivens,
            current_s_group_indices=current_s_group_indices,
            batch_size=batch_size,
            n_models=len(self.models),
            reset_free_energy_models=reset_free_energy_models,
            L_chirality_mask=L_chirality_mask,
            invert_chirality_function=invert_chirality_function,
        )

        # endregion

        losses = np.zeros((NO_epochs, len(self.models)))
        if calculate_test_loss:
            losses_test = np.zeros((NO_epochs, len(self.models)))

        logging.info("\nTraining the ensemble...\n")

        if wandb.config.free_energy_ensemble["early_stopping"]:
            best_losses = np.inf * np.ones(len(self.models))
            stopped = np.zeros(len(self.models), dtype=bool)
            no_improvement_counter = np.zeros(len(self.models), dtype=int)

            # TODO: Not a nice solution, but it works
            final_params = [torch.empty_like(item) for item in self.combined_params]

        for epoch in tqdm(
            range(starting_epoch, NO_epochs + starting_epoch),
            desc="Free energy training",
            mininterval=30,
        ):
            losses_per_model = self.train_epoch(
                dataloader_train,
                use_force_matching=False,
                only_get_test_loss=False,
            )

            if calculate_test_loss:
                test_losses_per_model = self.train_epoch(
                    dataloader_test,
                    use_force_matching=False,
                    only_get_test_loss=True,
                )

            if wandb.config.free_energy_ensemble["early_stopping"]:
                for i in range(len(self.models)):
                    if not stopped[i]:
                        if True:
                            if test_losses_per_model[i] < best_losses[i]:
                                best_losses[i] = test_losses_per_model[i]
                                no_improvement_counter[i] = 0
                            else:
                                no_improvement_counter[i] += 1

                            if no_improvement_counter[i] >= 50:
                                stopped[i] = True

                        else:
                            if test_losses_per_model[i] < 0.03:
                                stopped[i] = True

                                for j in range(len(final_params)):
                                    final_params[j][i][...] = self.combined_params[j][
                                        i
                                    ][...].clone()

                if np.all(stopped):
                    logging.info(
                        "Early stopping for all free energy models. Stopping training of the ensemble at epoch "
                        + str(epoch)
                    )
                    break

            if not plot_training_curve_as_whole:
                for loss, name in zip(
                    losses_per_model,
                    ["train_" + str(i) for i in range(len(self.models))],
                ):
                    wandb.log(
                        {name: loss},
                        step=epoch,
                    )

                if calculate_test_loss:
                    for loss, name in zip(
                        test_losses_per_model,
                        ["test_" + str(i) for i in range(len(self.models))],
                    ):
                        wandb.log(
                            {name: loss},
                            step=epoch,
                        )

            if wandb.config.free_energy_ensemble["early_stopping"]:
                losses_per_model[stopped] = np.nan
                test_losses_per_model[stopped] = np.nan

            losses[epoch - starting_epoch, :] = losses_per_model
            if calculate_test_loss:
                losses_test[epoch - starting_epoch, :] = test_losses_per_model

        if wandb.config.free_energy_ensemble["early_stopping"]:
            # for i in range(len(self.models)):
            #    if not stopped[i]:
            #        for j in range(len(final_params)):
            #            final_params[j][i][...] = self.combined_params[j][i][
            #                ...
            #            ].clone()

            # for i in range(len(self.combined_params)):
            #    self.combined_params[i].requires_grad = False
            #    self.combined_params[i][...] = final_params[i][...]
            #    self.combined_params[i].requires_grad = True

            self.combined_params = final_params

        if plot_training_curve_as_whole:
            epoch_range = np.arange(0, NO_epochs, dtype=int)

            for current_losses, name in zip(
                [losses, losses_test] if calculate_test_loss else [losses],
                ["train", "test"] if calculate_test_loss else ["train"],
            ):
                fig = plt.figure()

                for i in range(len(self.models)):
                    plt.plot(epoch_range, current_losses[:, i], label=f"model {i}")

                plt.legend()
                plt.xlabel("epoch")
                plt.ylabel("loss")
                # plt.yscale("log")
                plt.grid(True)

                """
                # Annotations:

                # Calculate the positions of the five equidistant points
                key_points = np.linspace(0, len(epoch_range) - 1, 5, dtype=int)
                key_epochs = epoch_range[key_points]
                key_losses = current_losses[key_points]

                # Add text annotations to these key points
                for i, (ep, ls) in enumerate(zip(key_epochs, key_losses)):
                    plt.gca().annotate(
                        f"{ls:.2f}",
                        (ep, ls),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                    )
                """

                if plot_output_dir is None:
                    save_fig_with_dpi(
                        fig,
                        dpi=300,
                        name="free_energy_loss_" + name,
                        epoch=current_epoch,
                    )
                else:
                    fig.savefig(
                        os.path.join(
                            plot_output_dir,
                            "free_energy_loss_" + name + f"_epoch_{current_epoch}.png",
                        ),
                        dpi=300,
                    )

    def rejection_sampling_in_range_1D(
        self, min_s: float, max_s: float, NO_samples: int
    ) -> torch.Tensor:
        """Sample from the ensemble using rejection sampling in the given range.
        This is used for the 1D backmapping plots only.

        Args:
            min_s (float): Minimum value of s.
            max_s (float): Maximum value of s.
            NO_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Sampled points.
        """

        with torch.no_grad():
            # Estimate the minimum potential energy
            s = torch.linspace(min_s, max_s, 100).unsqueeze(1).to("cuda")
            means = self.mean(s)
            min_energy = torch.min(means) - 0.5

            sampled_points = rejection_sampling(
                NO_samples,
                self.mean,
                min_energy,
                [[min_s, max_s]],
                beta=1.0,  # free energy values are already in units of kT
                use_torch_cuda=True,
            )

        return sampled_points[:, 0]

    def save_state(self, path: str):
        """Save the states of the ensemble's models and optimizers in a single file.

        Args:
            path (str): Path to the file.
        """

        # Before saving the state, get the individual models back from the combined model

        states = {
            "combined_params": self.combined_params,
            "combined_buffers": self.combined_buffers,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler is not None
            else None,
        }

        torch.save(states, path)

    def load_state(self, path: str):
        """Load the states of the ensemble's models, optimizer, and lr scheduler from a single file.
        Make sure to first add the models to the ensemble before loading the state.

        Args:
            path (str): Path to the file.
        """

        states = torch.load(path)

        if not "models" in states:
            # self.combined_model should already be set properly
            self.combined_params = states["combined_params"]
            self.combined_buffers = states["combined_buffers"]

        else:  # Only for backward compatibility # TODO: At some point this can be removed
            for model, state in zip(self.models, states["models"]):
                model.load_state_dict(state)

            self.combined_model = None
            self.combined_params = None
            self.combined_buffers = None

            self.combine_models(self.models)

        if (
            self.lr_scheduler is not None
            and "lr_scheduler" in states
            and states["lr_scheduler"] is not None
        ):
            self.lr_scheduler.load_state_dict(states["lr_scheduler"])

        if (
            self.optimizer is not None
            and "optimizer" in states
            and states["optimizer"] is not None
        ):
            self.optimizer.load_state_dict(states["optimizer"])

