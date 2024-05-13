import os
import wandb
import subprocess
import torch
from main.utils.matplotlib_helpers import set_default_paras
from main.utils.prepare_training import prepare_training
import logging
import argparse
from glob import glob
import yaml
from main.utils.wandb_helpers import merge_configs_recursive

if __name__ == "__main__":
    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=False, type=str, default="./configs/mb_AL.yaml"
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as file:
        _new_config_dict = yaml.safe_load(file)
        new_config_dict = {}
        for key, value in _new_config_dict.items():
            new_config_dict[key] = value["value"]  # wandb format

    do_resume = new_config_dict["do_resume"]

    if not do_resume:
        wandb.init(
            project="AL_CG_linear",
            notes=new_config_dict["target_system"]["name"]
            + (
                ("; " + new_config_dict["notes"])
                if ("notes" in new_config_dict and new_config_dict["notes"] is not None)
                else ""
            ),
            tags=[],
            mode="online",
            config=new_config_dict,
        )

    else:
        api = wandb.Api()
        run = api.run(f"AL_CG_linear/{new_config_dict['wandb_id_resume']}")
        old_config = run.config  # This is a dictionary of the config variables

        wandb_main_dir = os.environ.get("WANDB_DIR", ".") + "/wandb"
        # Get the run dir from the id
        wandb_run_dirs = glob(
            os.path.join(wandb_main_dir, f"run-*-{new_config_dict['wandb_id_resume']}")
        )
        assert len(wandb_run_dirs) == 1
        old_wandb_run_dir = wandb_run_dirs[0] + "/files"

        wandb.init(
            project="AL_CG_linear",
            notes="Resuming run "
            + str(new_config_dict["wandb_id_resume"])
            + "; "
            + new_config_dict["target_system"]["name"]
            + "; "
            + new_config_dict["notes"],
            tags=[],
            mode="online",
        )

        merged_config = merge_configs_recursive(old_config, new_config_dict)
        wandb.config.update(merged_config)

    ##############################

    if wandb.config.flow_architecture["use_float_64"]:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    # Print the slurm job id
    if "SLURM_JOB_ID" in os.environ:
        print("SLURM_JOB_ID: " + os.environ["SLURM_JOB_ID"])
        if not do_resume:
            wandb.config.slurm_job_id = os.environ["SLURM_JOB_ID"]
        else:
            wandb.config.update(
                {"slurm_job_id": os.environ["SLURM_JOB_ID"]}, allow_val_change=True
            )

    set_default_paras(double_width=True)  # matplotlib default paras

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        git_revision_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        )
    except Exception as e:
        git_revision_hash = "No git hash available."

    if not do_resume:
        wandb.config.git_hash = git_revision_hash
    else:
        wandb.config.update({"git_hash": git_revision_hash}, allow_val_change=True)

    if not do_resume:
        wandb.config.run_dir = os.path.basename(os.path.dirname(wandb.run.dir))
    else:
        wandb.config.update(
            {"run_dir": os.path.basename(os.path.dirname(wandb.run.dir))},
            allow_val_change=True,
        )

    os.system(
        "mkdir -p " + os.path.join(wandb.run.dir, "models")
    )  # for saving models later
    os.system("mkdir -p " + os.path.join(wandb.run.dir, "additional"))

    AL_manager = prepare_training()  # Prepare AL manager according to wandb config

    if do_resume:
        checkpoint_dir = os.path.join(
            old_wandb_run_dir,
            f"checkpoints/epoch_{new_config_dict['wandb_checkpoint_index_resume']}",
        )

        (
            new_points_training,
            new_points_test,
            new_points_training_group_indices,
            new_points_test_group_indices,
        ) = AL_manager.load_checkpoint(checkpoint_dir)

        # The new points have not been added to the ALDataset yet, therefore:
        if not (
            AL_manager.current_iteration == 1
            and AL_manager.current_epoch_in_iteration == 0
        ):  # In case we were training by example before, do not add them, they already were added
            if wandb.config.flow_architecture["filtering_type"] == "mirror":
                new_points_training = torch.cat(
                    [
                        new_points_training,
                        AL_manager.system.invert_cg_configurations(new_points_training),
                    ],
                    dim=0,
                )
                new_points_training_group_indices = torch.cat(
                    [
                        new_points_training_group_indices,
                        new_points_training_group_indices
                        + torch.max(new_points_training_group_indices)
                        + 1,
                    ],
                    dim=0,
                )

                new_points_test = torch.cat(
                    [
                        new_points_test,
                        AL_manager.system.invert_cg_configurations(new_points_test),
                    ],
                    dim=0,
                )
                new_points_test_group_indices = torch.cat(
                    [
                        new_points_test_group_indices,
                        new_points_test_group_indices
                        + torch.max(new_points_test_group_indices)
                        + 1,
                    ],
                    dim=0,
                )

            AL_manager.AL_dataset.add_high_error_points(
                new_points_training,
                new_points_training_group_indices,
                put_on_train=True,
                iteration=AL_manager.current_iteration - 1,
            )
            AL_manager.AL_dataset.add_high_error_points(
                new_points_test,
                new_points_test_group_indices,
                put_on_train=False,
                iteration=AL_manager.current_iteration - 1,
            )

        AL_manager.initialize_lr_scheduler()

    while True:
        stopped = AL_manager.train_epoch()  # Didn't find enough high-error points?

        if stopped:
            logging.info("Stopped.")
            break

    wandb.finish()
