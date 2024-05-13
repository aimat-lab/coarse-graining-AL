from main.utils.plot_free_energy import plot_free_energy_histogram
import matplotlib.pyplot as plt
import numpy as np
from mdtraj.geometry.dihedral import (
    _dihedral as mdtraj_dihedral,
)  # See https://en.wikipedia.org/wiki/Dihedral_angle for definition
import mdtraj
import os
from main.systems.mb.mb_system import plot_1D_free_energy_from_histogram
from main.utils.matplotlib_helpers import set_content_to_rasterized

aldp_dihedral_indices = {
    "phi": [14, 8, 6, 4],
    "psi": [6, 8, 14, 16],
    "theta_1": [1, 4, 6, 8],
    "theta_2": [18, 16, 14, 8],
    "dih_methyl": [4, 6, 8, 10],  # methyl dihedral
    "dih_oxygen_1": [8, 6, 4, 5],  # dihedral of first oxygen
    "dih_oxygen_2": [16, 8, 14, 15],  # dihedral of second oxygen
}

aldp_dihedral_plots = [
    ["phi", "psi"],
    ["phi"],
    ["psi"],
    ["theta_1"],
    ["theta_2"],
    ["dih_methyl"],
    ["phi", "dih_methyl"],
    ["psi", "dih_methyl"],
    ["dih_oxygen_1"],
    ["dih_oxygen_2"],
    ["theta_1", "dih_methyl"],
    ["theta_2", "dih_methyl"],
]
aldp_dihedral_plots_without_CG = aldp_dihedral_plots[3:]


def create_dihedral_plot_2D(
    npy_data_or_path,
    image_path,
    angle_indices_a,
    angle_indices_b,
    angle_label_a,
    angle_label_b,
    weights=None,
    vmax=11.0,
    plot_args={},
):
    if "width" in plot_args and "height" in plot_args:
        plt.figure(figsize=(plot_args["width"], plot_args["height"]))
    else:
        plt.figure()

    if isinstance(npy_data_or_path, str):
        trajectory_data = np.load(npy_data_or_path)
    else:
        trajectory_data = npy_data_or_path

    traj_from_npy = mdtraj.Trajectory(trajectory_data, None)

    dihedrals_a = mdtraj_dihedral(
        traj_from_npy, np.array(angle_indices_a)[None, :], periodic=False
    )
    # Convert to degrees
    dihedrals_a = np.degrees(dihedrals_a)[:, 0]

    dihedrals_b = mdtraj_dihedral(
        traj_from_npy, np.array(angle_indices_b)[None, :], periodic=False
    )
    # Convert to degrees
    dihedrals_b = np.degrees(dihedrals_b)[:, 0]

    if True:
        fig = plot_free_energy_histogram(
            dihedrals_a,
            dihedrals_b,
            ax=plt.gca(),
            vmax=vmax,
            weights=weights,
            nbins=100,
            cbar=not (
                "disable_colorbar" in plot_args and plot_args["disable_colorbar"]
            ),
        )[0]

        set_content_to_rasterized(fig)
    else:
        plt.scatter(dihedrals_a, dihedrals_b, s=0.1, c=weights)

    if not "disable_labels" in plot_args or not plot_args["disable_labels"]:
        plt.xlabel(angle_label_a)
        plt.ylabel(angle_label_b)

    plt.xticks([-90, 0, 90])
    plt.yticks([-90, 0, 90])

    plt.tight_layout()
    plt.savefig(image_path, dpi=600, bbox_inches="tight")


def create_dihedral_plot_1D(
    npy_data_or_path, image_path, angle_indices, angle_label, weights=None
):
    plt.figure()

    if isinstance(npy_data_or_path, str):
        trajectory_data = np.load(npy_data_or_path)
    else:
        trajectory_data = npy_data_or_path

    traj_from_npy = mdtraj.Trajectory(trajectory_data, None)

    dihedrals = mdtraj_dihedral(
        traj_from_npy, np.array(angle_indices)[None, :], periodic=False
    )
    # Convert to degrees
    dihedrals = np.degrees(dihedrals)[:, 0]

    fig = plot_1D_free_energy_from_histogram(
        dihedrals[:, None], ax=plt.gca(), weights=weights
    )

    plt.xlabel(angle_label)

    set_content_to_rasterized(fig)

    plt.tight_layout()
    plt.savefig(image_path, dpi=600, bbox_inches="tight")


def replace_with_latex_symbol(label):
    if label == "phi":
        return r"$\phi$"
    elif label == "psi":
        return r"$\psi$"
    elif label == "theta_1":
        return r"$\theta_1$"
    elif label == "theta_2":
        return r"$\theta_2$"
    else:
        return None


def plot_all_dihedral_maps(
    npy_data_or_path,
    output_dir,
    weights=None,
    vmax=11.0,
    without_CG=False,
    plot_args={},
):
    if isinstance(npy_data_or_path, str):
        trajectory_data = np.load(npy_data_or_path)
    else:
        trajectory_data = npy_data_or_path

    for dihedral_plot in (
        aldp_dihedral_plots_without_CG if without_CG else aldp_dihedral_plots
    ):
        label_a = dihedral_plot[0]
        label_a_replaced = replace_with_latex_symbol(label_a)
        label_a = label_a_replaced if label_a_replaced is not None else label_a

        if len(dihedral_plot) == 2:
            label_b = dihedral_plot[1]
            label_b_replaced = replace_with_latex_symbol(label_b)
            label_b = label_b_replaced if label_b_replaced is not None else label_b

        if len(dihedral_plot) == 1:
            create_dihedral_plot_1D(
                trajectory_data,
                os.path.join(output_dir, f"{dihedral_plot[0]}.pdf"),
                aldp_dihedral_indices[dihedral_plot[0]],
                label_a,
                weights=weights,
            )
        elif len(dihedral_plot) == 2:
            create_dihedral_plot_2D(
                trajectory_data,
                os.path.join(output_dir, f"{dihedral_plot[0]}_{dihedral_plot[1]}.pdf"),
                aldp_dihedral_indices[dihedral_plot[0]],
                aldp_dihedral_indices[dihedral_plot[1]],
                label_a,
                label_b,
                weights=weights,
                vmax=vmax,
                plot_args=plot_args,
            )
        else:
            raise ValueError(f"Invalid dihedral plot {dihedral_plot}")
