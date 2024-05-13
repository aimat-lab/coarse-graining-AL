""" This script can be used to re-run the analysis for a single checkpoint after a training experiment is finished.
"""

import argparse
import wandb
from main.utils.matplotlib_helpers import set_default_paras
import os
import torch
from glob import glob
from main.utils.prepare_training import prepare_training
import main.utils.matplotlib_helpers as defaults
import logging
from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble

plot_args_aldp = {
    "disable_colorbar": True,
    "disable_labels": True,
    "display_new_points": True,
    "display_training_points": False,
    "width": defaults.column_width / 2.0 + 0.25,  # fit 4 plots in one row
    # "width": (defaults.column_width / 2.0 + 0.25) * 2,  # fit 2 plots in one row
    "height": 1.3,
    # "height": 2.6, # fit 2 plots in one row
    "plot_MC_starting_points": False,
}

plot_args_mb = {
    "width": defaults.column_width / 2.0 + 0.35,  # fit 4 plots in one row
    "height": 2.6,
    "countour_linewidths": 0.5,
    "plot_MC_starting_points": False,
    "do_plot_2D_training_points": False,
    "save_PMF_to_file": False,
}


def run_analysis_for_checkpoint(wandb_dir, checkpoint_epoch_indices=None):
    # Get the wandb run id from wandb_run_dir
    wandb_run_id = wandb_dir.strip("/").split("/")[-1].split("-")[-1]
    wandb_run_dir = os.path.join(args.wandb_dir, "files")

    # Resume run to get config, but don't sync anything
    api = wandb.Api()
    run = api.run(f"AL_CG_linear/{wandb_run_id}")
    old_config = run.config  # This is a dictionary of the config variables

    # Initialize a new run disabled, just to get the config
    wandb.init(project="AL_CG_linear", mode="disabled")

    # Manually set the config variables from the old run
    wandb.config.update(old_config)

    if wandb.config.target_system["name"] == "aldp":
        plot_args = plot_args_aldp
    elif wandb.config.target_system["name"] == "mb":
        plot_args = plot_args_mb
    else:
        raise ValueError(f"Unknown target system {wandb.config.target_system['name']}.")

    if wandb.config.flow_architecture["use_float_64"]:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    AL_manager = prepare_training()  # Prepare AL manager according to wandb config

    if checkpoint_epoch_indices is None:
        all_checkpoint_dirs = glob(os.path.join(wandb_run_dir, "checkpoints/epoch_*"))
        checkpoint_epoch_indices = [
            int(checkpoint_dir.split("_")[-1]) for checkpoint_dir in all_checkpoint_dirs
        ]
        checkpoint_epoch_indices = sorted(checkpoint_epoch_indices)

    for checkpoint_epoch_index in checkpoint_epoch_indices:
        checkpoint_dir = os.path.join(
            wandb_run_dir, f"checkpoints/epoch_{checkpoint_epoch_index}"
        )
        AL_manager.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
        )

        additional = torch.load(os.path.join(checkpoint_dir, "additional.pt"))

        display_training_points = additional["display_training_points"]
        plot_predicted_F = additional["plot_predicted_F"]
        s_samples = additional["s_samples"]
        n_bins_backmapping = additional["n_bins_backmapping"]

        new_points_training = additional["new_points_training"]
        new_points_training_group_indices = additional[
            "new_points_training_group_indices"
        ]
        new_points_test = additional["new_points_test"]
        new_points_test_group_indices = additional["new_points_test_group_indices"]

        MC_starting_points = additional.get("MC_starting_points", None)

        set_default_paras(width_height=(plot_args["width"], plot_args["height"]))

        ##### Recreate ensemble to set the correct standard scalers #####

        # Initialize the ensemble with models
        ensemble = FreeEnergyEnsemble(
            output_force=False,
            learning_rate=wandb.config.free_energy_ensemble["learning_rate"],
        )

        if (
            wandb.config.free_energy_ensemble["apply_mean_standard_scaler"]
            or wandb.config.free_energy_ensemble["apply_std_standard_scaler"]
        ):
            NO_evaluations = AL_manager.AL_dataset.get_current_evaluation_counters()[
                :, 0
            ]
            cg_configs = AL_manager.AL_dataset.get_current_s()
            standard_scaler_dataset = cg_configs[NO_evaluations > 0]

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

        AL_manager.AL_ensemble = ensemble

        AL_manager.AL_ensemble.load_state(os.path.join(checkpoint_dir, "ensemble.pt"))

        ##########

        AL_manager.create_checkpoint(
            new_points_training=new_points_training,
            new_points_training_group_indices=new_points_training_group_indices,
            new_points_test=new_points_test,
            new_points_test_group_indices=new_points_test_group_indices,
            MC_starting_points=MC_starting_points,
            silence_stopped_print=True,
            calculate_metrics=True,
            s_samples=s_samples,
            plot_predicted_F=plot_predicted_F,
            display_training_points=display_training_points,
            do_not_save=True,
            print_instead_of_logging=True,
            analysis_output_dir=os.path.join(
                wandb_run_dir, "checkpoints", f"epoch_{checkpoint_epoch_index}"
            ),
            n_bins_backmapping=n_bins_backmapping,
            plot_args=plot_args,
        )

    # Avoid error message when exiting
    if hasattr(AL_manager.system, "target"):
        AL_manager.system.target.pool.terminate()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run analysis for checkpoints of a training experiment."
    )
    parser.add_argument(
        "--wandb_dir", type=str, help="Path to wandb run dir", required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epoch indices of the checkpoints to analyse",
        required=False,
        default=None,  # indicates that all checkpoints should be analysed
        nargs="*",
    )
    args = parser.parse_args()

    run_analysis_for_checkpoint(
        wandb_dir=args.wandb_dir,
        checkpoint_epoch_indices=args.epochs
        if args.epochs is not None
        else None,  # If no epoch is specified, analyse all checkpoints
    )
