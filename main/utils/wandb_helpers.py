import os
from glob import glob
import time
import sys
import io
import wandb
import PIL.Image as Image


def merge_configs_recursive(old, new):
    """
    Recursively merges the new configuration into the old one.
    """

    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in new.items():
            if key in old:
                old[key] = merge_configs_recursive(old[key], value)
            else:
                old[key] = value
        return old
    return new


def preserve_log(run_files_dir: str, run_counter: int):
    """Copy the old output.log to output-<run_counter>.log before calling
    `wandb.finish()`. This makes sure that when resuming the current run at a
    later point, the old log will not be overwritten.

    Args:
        run_files_dir (str): `files` subdirectory of the current wandb run.
        run_counter (int): Current run counter.
    """

    print(flush=True)
    sys.stdout.flush()
    time.sleep(2)

    if len(glob(os.path.join(run_files_dir, "output.log"))) == 0:
        return  # No log file present that might be overwritten later

    os.system(
        "cp "
        + os.path.join(run_files_dir, "output.log")
        + " "
        + os.path.join(run_files_dir, f"output_{run_counter}.log")
    )


def save_fig_with_dpi(fig, dpi: int, name: str, epoch: int = None):
    """Save a matplotlib figure with a given dpi to wandb.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        dpi (int): Dots per inch.
        name (str): Name of the figure.
        epoch (int, optional): Current epoch. Defaults to None.
    """

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)  # bbox_inches="tight")
    buf.seek(0)
    wandb.log(
        {
            name: wandb.Image(Image.open(buf)),
        },
        step=epoch,
    )
