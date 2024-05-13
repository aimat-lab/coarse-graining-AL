from main.systems.aldp.dihedral_utils import plot_all_dihedral_maps
from main.utils.matplotlib_helpers import set_default_paras
import numpy as np
import main.utils.matplotlib_helpers
import os

if __name__ == "__main__":
    set_default_paras()

    plot_args = {
        "disable_colorbar": True,
        "disable_labels": True,
        "width": main.utils.matplotlib_helpers.column_width / 2.0
        + 0.25,  # fit 4 plots in one row
        "height": 1.3,
    }

    # Analyze dihedrals of ground truth trajectory

    dataset = np.load("../datasets/test_cartesian.npy")

    # Create the directory if it does not exist
    os.makedirs("./dihedral_maps/fab_test", exist_ok=True)

    plot_all_dihedral_maps(
        dataset,
        "./dihedral_maps/fab_test",
        vmax=25.0,
        plot_args=plot_args,
    )
