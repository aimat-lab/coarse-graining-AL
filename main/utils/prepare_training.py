import wandb
import logging
from main.models.RNVP_flows import create_conditional_RNVP_flow
from main.systems.aldp.coord_trafo import create_coord_trafo
from main.systems.aldp.create_model import create_RQS_flow
import numpy as np
import torch
from main.utils.FastTensorDataLoader import FastTensorDataLoader
from main.active_learning.active_learning_dataset import ActiveLearningDataset
from main.active_learning.active_learning_managers import EnsembleALManager
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble
from main.systems.mb.mb_system import MBSystem
from main.systems.aldp.aldp_system import AldpSystem
from main.active_learning.samplers.active_learning_sampler_MC import (
    ActiveLearningSamplerMC,
)
import matplotlib.pyplot as plt
from main.utils.plot_free_energy import plot_free_energy_histogram
import os
from main.utils.matplotlib_helpers import set_default_paras
from main.models.gaussian_flow_1D import GaussianFlow1D
from normflows.distributions.base import UniformGaussian


def prepare_training() -> EnsembleALManager:
    # region Construct starting dataloader

    if wandb.config.target_system["name"] == "aldp":
        ground_truth_dataset = np.load(wandb.config.ground_truth_dataset_path)
        ground_truth_dataset = ground_truth_dataset.reshape(-1, 66)
        ground_truth_dataset *= 0.1  # angstrom to nm

        starting_dataset = np.load(wandb.config.starting_dataset_path)
        starting_dataset = starting_dataset.reshape(-1, 66)
        starting_dataset *= 0.1  # angstrom to nm

        starting_dataset = starting_dataset[
            :: wandb.config.stride_N_samples_starting_dataset
        ]
        starting_dataset = starting_dataset[
            : wandb.config.use_N_samples_starting_dataset
        ]

        coords_tensor_starting = torch.tensor(
            starting_dataset, dtype=torch.get_default_dtype()
        )

        if wandb.config.flow_architecture["filtering_type"] == "mirror":
            coords_tensor_starting_inverted = -1.0 * coords_tensor_starting
            coords_tensor_starting = torch.cat(
                [coords_tensor_starting, coords_tensor_starting_inverted], dim=0
            )

    elif wandb.config.target_system["name"] == "mb":
        starting_dataset = np.load(wandb.config.starting_dataset_path)

        # Shuffle with seed
        np.random.seed(123)
        np.random.shuffle(starting_dataset)

        starting_dataset = np.unique(starting_dataset, axis=0)

        starting_dataset = starting_dataset[
            -wandb.config.use_N_samples_starting_dataset :
        ]
        coords_tensor_starting = torch.tensor(
            starting_dataset, dtype=torch.get_default_dtype()
        )
    else:
        raise Exception(
            f"Target system {wandb.config.target_system['name']} not implemented."
        )

    starting_dataloader = FastTensorDataLoader(
        coords_tensor_starting,
        batch_size=wandb.config.batch_size_example,
        shuffle=True,
        drop_last=True,
    )

    # endregion

    # Create the target system
    if wandb.config.target_system["name"] == "mb":
        system = MBSystem(beta=wandb.config.beta)
    elif wandb.config.target_system["name"] == "aldp":
        ground_truth_dataset_tensor = torch.tensor(ground_truth_dataset).to(
            dtype=torch.get_default_dtype()
        )
        trafo = create_coord_trafo(
            # phi_shift=wandb.config.flow_architecture["phi_shift"],
            phi_shift=wandb.config.flow_architecture.get("phi_shift", 0),
            # psi_shift=wandb.config.flow_architecture["psi_shift"],
            psi_shift=wandb.config.flow_architecture.get("psi_shift", 0),
        )
        system = AldpSystem(
            60,
            wandb.config.CG_indices,
            trafo,
            ground_truth_trajectory=ground_truth_dataset_tensor,
        )
    else:
        raise Exception(
            f"Target system {wandb.config.target_system['name']} not implemented."
        )

    if wandb.config.target_system["name"] == "mb":
        if wandb.config.flow_architecture["type"] == "RNVP":
            # Use a conditional invertible neural network with variable substitution strategy:
            cond_flow = create_conditional_RNVP_flow(
                ndim_in=1,
                ndim_cond=1,
                subnets_dimensionality=wandb.config.flow_architecture[
                    "subnets_dimensionality"
                ],
                NO_blocks=wandb.config.flow_architecture["NO_blocks"],
                seed=wandb.config.get("INN_seed"),
            )

            if wandb.config.flow_architecture["conditioning_standard_scaler"]:
                raise Exception(
                    "Standard scaler not implemented for RNVP flow architecture."
                )
        elif wandb.config.flow_architecture["type"] == "simple_1D":
            base = UniformGaussian(1, ind=[])  # Gaussian base distribution
            base = base.cuda()

            cond_flow = GaussianFlow1D(
                q0=base,
                subnet_dimensions=[
                    1,
                ]
                + wandb.config.flow_architecture["subnets_dimensionality"]
                + [
                    1,
                ],
                standard_scaler_dataset=system.cartesian_to_internal(
                    coords_tensor_starting
                )[:, system.CG_mask]
                if wandb.config.flow_architecture["conditioning_standard_scaler"]
                else None,
            )
            cond_flow = cond_flow.cuda()

        else:
            raise Exception(
                f"Flow architecture {wandb.config.flow_architecture['type']} not implemented."
            )
    else:
        cond_flow = create_RQS_flow(
            trafo,
            conditional=True,
            conditional_indices=wandb.config.CG_indices,
            periodic_conditioning=wandb.config.flow_architecture[
                "periodic_conditioning"
            ],
            use_fab_periodic_conditioning=wandb.config.flow_architecture[
                "use_fab_periodic_conditioning"
            ],
            use_cos_sin_periodic_representation_identity=wandb.config.flow_architecture[
                "use_cos_sin_periodic_representation_identity"
            ],
        )
        cond_flow = cond_flow.to("cuda")

        if wandb.config.flow_architecture["conditioning_standard_scaler"]:
            raise Exception(
                "Standard scaler not implemented for RQS flow architecture."
            )

    logging.info(
        "Number of parameters in flow: "
        + str(sum([np.prod(p.size()) for p in cond_flow.parameters()])),
    )

    if wandb.config.flow_architecture["optimizer"] == "adam":
        INN_optimizer = torch.optim.Adam(
            [p for p in cond_flow.parameters() if p.requires_grad],
            lr=wandb.config.lr_example,
        )
    elif wandb.config.flow_architecture["optimizer"] == "sgd":
        INN_optimizer = torch.optim.SGD(
            [p for p in cond_flow.parameters() if p.requires_grad],
            lr=wandb.config.lr_example,
        )
    else:
        raise Exception(
            f"Optimizer {wandb.config.flow_architecture['flow_optimizer']} not implemented."
        )

    AL_dataset = ActiveLearningDataset(
        cond_flow=cond_flow,
        starting_size=wandb.config.starting_size_AL_dataset,
        batch_size=wandb.config.batch_size_probability,
        test_dataset_fraction=wandb.config.test_dataset_fraction,
    )

    # Initialize the ensemble with models
    ensemble = FreeEnergyEnsemble(
        output_force=False,
        learning_rate=wandb.config.free_energy_ensemble[
            "learning_rate"
        ],  # Since we are currently not using conventional force-matching
    )

    if (
        wandb.config.free_energy_ensemble["apply_mean_standard_scaler"]
        or wandb.config.free_energy_ensemble["apply_std_standard_scaler"]
    ):
        standard_scaler_dataset = system.cartesian_to_internal(coords_tensor_starting)[
            :, system.CG_mask
        ]

    ensemble.add_FC_free_energy_models(
        NO_models=wandb.config.free_energy_ensemble["ensemble_size"],
        size_of_latent_space=len(wandb.config.CG_indices),
        hidden_layers=wandb.config.free_energy_ensemble["hidden_layers"],
        use_2pi_periodic_representation=wandb.config.free_energy_ensemble[
            "periodic_input_rep"
        ],
        standard_scaler=(
            torch.mean(standard_scaler_dataset, dim=0, keepdim=True).item()
            if wandb.config.free_energy_ensemble["apply_mean_standard_scaler"]
            else 0.0,
            torch.std(
                standard_scaler_dataset, dim=0, keepdim=True, unbiased=False
            ).item()
            if wandb.config.free_energy_ensemble["apply_std_standard_scaler"]
            else 1.0,
        ),
    )

    # Initialize the sampler
    sampler = ActiveLearningSamplerMC(
        ensemble,
        step_size=wandb.config.MC_sampling["stepsize"],
        error_threshold=wandb.config.MC_sampling["error_threshold"],
        periodic_wrapping_ranges=None
        if wandb.config.target_system["name"] == "mb"
        else system.get_uniform_CG_ranges(),
    )

    AL_manager = EnsembleALManager(
        cond_flow=cond_flow,
        cond_flow_optimizer=INN_optimizer,
        system=system,
        AL_ensemble=ensemble,
        AL_dataset=AL_dataset,
        AL_sampler=sampler,
        starting_dataloader=starting_dataloader,
    )

    return AL_manager

