from main.systems.aldp.simulation.initialize_simulation import (
    initialize_simulation_aldp,
)
from simtk import unit
import torch
import numpy as np
import boltzgen as bg
import os
from mdtraj.geometry.dihedral import (
    _dihedral as mdtraj_dihedral,
)
from main.systems.aldp.dihedral_utils import aldp_dihedral_indices
import mdtraj
from mdtraj.geometry.dihedral import (
    _dihedral as mdtraj_dihedral,
)  # See https://en.wikipedia.org/wiki/Dihedral_angle for definition
from typing import List
import logging
import matplotlib.pyplot as plt


def get_minimum_energy_configuration(
    overwrite_cache=False, minimum_energy_configuration_path=None
):
    if minimum_energy_configuration_path is None:
        cache_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "simulation/input_files/minimum_energy_configuration.pt",
        )
    else:
        cache_path = minimum_energy_configuration_path

    default_dtype = torch.get_default_dtype()

    if os.path.exists(cache_path) and not overwrite_cache:
        position_tensor = torch.load(cache_path)
        logging.info("Using cached minimum energy configuration")
        return position_tensor.to(dtype=default_dtype)

    sim = initialize_simulation_aldp(constraints=None, platform="CUDA")

    sim.minimizeEnergy()
    state = sim.context.getState(getPositions=True)
    position = state.getPositions(True).value_in_unit(unit.nanometer)

    position_tensor = torch.tensor(position.reshape(1, 66)).to(dtype=default_dtype)

    torch.save(position_tensor, cache_path)

    return position_tensor


def create_coord_trafo(
    transformation_data=None,
    default_std={"bond": 0.005, "angle": 0.15, "dih": 0.2},
    phi_shift=0,
    psi_shift=0,
    minimum_energy_configuration_path=None,
):
    """Create coordinate transformation for alanine dipeptide

    Args:
        transformation_data (torch.tensor, optional): Tensor containing the transformation data used to setup the means and standard deviations of the coordinates.
            If None, the minimum energy configuration is used. Defaults to None.
        default_std (dict, optional): Dictionary containing the default standard deviations for the different coordinate types.
            Defaults to {"bond": 0.005, "angle": 0.15, "dih": 0.2}. The means are taken from the minimum energy configuration.
        phi_shift (float, optional): Shift the phi angle by this amount. Defaults to 0.
        psi_shift (float, optional): Shift the psi angle by this amount. Defaults to 0.
        minimum_energy_configuration_path (str, optional): Path to the minimum energy configuration. If None, the default path is used.
    """

    # Generate internal coordinate transformation for alanine dipeptide
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),  # phi
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (9, [8, 6, 4]),
        (10, [8, 6, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),  # psi
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19]),  # 19 in total
    ]  # Those are 19*3 = 57 internal DOF. Since we are not using the PCA, there will be an additional
    # three "last internal coordinates" added to the 57, resulting in 60 internal coordinates.

    cart_indices = [
        8,
        6,
        14,
    ]  # Originally the indices used for the PCA, but we are not using PCA here
    ind_circ_dih = [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 15, 16]  # circular coordinates
    ndim = 66

    if transformation_data is None:
        minimum_energy_config = get_minimum_energy_configuration(
            minimum_energy_configuration_path=minimum_energy_configuration_path
        )

    coordinate_transform = bg.flows.CoordinateTransform(
        minimum_energy_config
        if transformation_data is None
        else transformation_data,  # used to setup the means of the coordinates
        ndim,
        z_matrix,
        cart_indices,
        mode="internal",
        ind_circ_dih=ind_circ_dih,  # treated as circular coordinates
        shift_dih=False,
        default_std=default_std,
    )

    if phi_shift != 0:
        # phi is at index 0

        coordinate_transform.transform.ic_transform.mean_dih[0] += (
            phi_shift / 180.0 * np.pi
        )
        # Wrap:
        coordinate_transform.transform.ic_transform.mean_dih[0] = (
            coordinate_transform.transform.ic_transform.mean_dih[0] + np.pi
        ) % (2 * np.pi) - np.pi

    if psi_shift != 0:
        # psi is at index 8

        coordinate_transform.transform.ic_transform.mean_dih[8] += (
            psi_shift / 180.0 * np.pi
        )
        # Wrap:
        coordinate_transform.transform.ic_transform.mean_dih[8] = (
            coordinate_transform.transform.ic_transform.mean_dih[8] + np.pi
        ) % (2 * np.pi) - np.pi

    return coordinate_transform


def get_trafo_output_info(trafo: bg.flows.CoordinateTransform):
    output_info = {}

    inds1 = trafo.transform.ic_transform.inds_for_atom[
        trafo.transform.ic_transform.rev_z_indices[:, 1]
    ]
    inds2 = trafo.transform.ic_transform.inds_for_atom[
        trafo.transform.ic_transform.rev_z_indices[:, 2]
    ]
    inds3 = trafo.transform.ic_transform.inds_for_atom[
        trafo.transform.ic_transform.rev_z_indices[:, 3]
    ]
    inds4 = trafo.transform.ic_transform.inds_for_atom[
        trafo.transform.ic_transform.rev_z_indices[:, 0]
    ]

    for i, index in enumerate(inds4[:, 0]):
        output_info[index.item()] = [
            "bond",
            [int(inds1[i][0].item() / 3), int(inds4[i][0].item() / 3)],
        ]

    for i, index in enumerate(inds4[:, 1]):
        output_info[index.item()] = [
            "angle",
            [
                int(inds2[i][0].item() / 3),
                int(inds1[i][0].item() / 3),
                int(inds4[i][0].item() / 3),
            ],
        ]

    for i, index in enumerate(inds4[:, 2]):
        output_info[index.item()] = [
            "dihedral",
            [
                int(inds3[i][0].item() / 3),
                int(inds2[i][0].item() / 3),
                int(inds1[i][0].item() / 3),
                int(inds4[i][0].item() / 3),
            ],
        ]

    output_keys = list(output_info.keys())
    cart_indices = [i for i in list(range(66)) if i not in output_keys]

    for i in cart_indices[0::3]:
        output_info[i] = ["cartesian", [int(i / 3)]]
        output_info[i + 1] = ["cartesian", [int(i / 3)]]
        output_info[i + 2] = ["cartesian", [int(i / 3)]]

    output_info_permuted = {}

    for index, value in output_info.items():
        output_info_permuted[index] = output_info[trafo.transform.permute[index].item()]

    # Now the first 9 cartesian coordinates are converted to 3 internal coordinates
    output_info_final = []

    output_info_final.append(
        [
            "bond",
            [output_info_permuted[0][1][0], output_info_permuted[3][1][0]],
        ]
    )
    output_info_final.append(
        [
            "bond",
            [output_info_permuted[0][1][0], output_info_permuted[6][1][0]],
        ]
    )
    output_info_final.append(
        [
            "angle",
            [
                output_info_permuted[3][1][0],
                output_info_permuted[0][1][0],
                output_info_permuted[6][1][0],
            ],
        ]
    )

    for i in range(9, 66):  # the rest are the previous internal coordinates
        output_info_final.append(output_info_permuted[i])

    return output_info_final


def transform_trajectory(
    tensor_data_or_path, trafo: bg.flows.CoordinateTransform, stride=10
):
    if isinstance(tensor_data_or_path, str):
        trajectory_data = torch.tensor(np.load(tensor_data_or_path).reshape(-1, 66))
    else:
        trajectory_data = tensor_data_or_path

    # np to torch
    default_dtype = torch.get_default_dtype()
    trajectory_data = trajectory_data.to(dtype=default_dtype)

    trajectory_data = trajectory_data[::stride, :]
    # trajectory_data = trajectory_data.to("cuda")

    transformed, _ = trafo.transform.forward(trajectory_data)  # 60 DOF

    # transformed_back, _ = trafo.transform.inverse(transformed) # transformed_back has 66 DOF again

    return transformed


def _get_all_dihedrals(traj_numpy):
    traj = mdtraj.Trajectory(traj_numpy, None)
    all_dihedrals = np.zeros((traj_numpy.shape[0], len(aldp_dihedral_indices.keys())))

    counter = 0
    for key, value in aldp_dihedral_indices.items():
        dihedrals = mdtraj_dihedral(
            traj, np.array(aldp_dihedral_indices[key])[None, :], periodic=False
        )

        all_dihedrals[:, counter] = dihedrals[:, 0]

        counter += 1

    return all_dihedrals


def unscale_internal_coordinates(
    x_int: torch.Tensor,
    trafo: bg.flows.CoordinateTransform,
    means_and_stds: torch.Tensor = None,
    indices: List[int] = None,
):
    if means_and_stds is None:
        means_and_stds = get_means_and_stds(trafo)

    means, stds, _, _, dih_indices = means_and_stds

    if indices is not None:
        means = means[indices]
        stds = stds[indices]

        # Only keep those dih_indices that are in indices:
        dih_indices = torch.tensor(
            [i for i, index in enumerate(indices) if index in dih_indices],
            dtype=torch.int64,
            device=x_int.device,
        )

    output = x_int * stds
    output += means
    output[:, dih_indices] = (output[:, dih_indices] + np.pi) % (2 * np.pi) - np.pi

    return output


def scale_internal_coordinates(
    x_int_unscaled: torch.Tensor,
    trafo: bg.flows.CoordinateTransform,
    means_and_stds: torch.Tensor = None,
    indices: List[int] = None,
):
    if means_and_stds is None:
        means_and_stds = get_means_and_stds(trafo)

    means, stds, _, _, dih_indices = means_and_stds

    if indices is not None:
        means = means[indices]
        stds = stds[indices]

        # Only keep those dih_indices that are in indices:
        dih_indices = torch.tensor(
            [i for i, index in enumerate(indices) if index in dih_indices],
            dtype=torch.int64,
            device=x_int_unscaled.device,
        )

    output = x_int_unscaled - means
    output[:, dih_indices] = (output[:, dih_indices] + np.pi) % (2 * np.pi) - np.pi
    output /= stds

    return output


def get_means_and_stds(trafo: bg.flows.CoordinateTransform):
    means_final = []
    stds_final = []

    means_final.extend(
        [
            trafo.transform.mean_b1.item(),
            trafo.transform.mean_b2.item(),
            trafo.transform.mean_angle.item(),
        ]
    )
    stds_final.extend(
        [
            trafo.transform.std_b1.item(),
            trafo.transform.std_b2.item(),
            trafo.transform.std_angle.item(),
        ]
    )

    means_temp = torch.zeros(66)
    stds_temp = torch.zeros(66)

    means_temp[
        trafo.transform.ic_transform.bond_indices
    ] = trafo.transform.ic_transform.mean_bonds
    stds_temp[
        trafo.transform.ic_transform.bond_indices
    ] = trafo.transform.ic_transform.std_bonds
    means_temp[
        trafo.transform.ic_transform.angle_indices
    ] = trafo.transform.ic_transform.mean_angles
    stds_temp[
        trafo.transform.ic_transform.angle_indices
    ] = trafo.transform.ic_transform.std_angles
    means_temp[
        trafo.transform.ic_transform.dih_indices
    ] = trafo.transform.ic_transform.mean_dih
    stds_temp[
        trafo.transform.ic_transform.dih_indices
    ] = trafo.transform.ic_transform.std_dih

    means_temp = means_temp[trafo.transform.permute]
    stds_temp = stds_temp[trafo.transform.permute]

    means_temp = means_temp[9:]
    stds_temp = stds_temp[9:]

    means_final.extend(means_temp.numpy().tolist())
    stds_final.extend(stds_temp.numpy().tolist())

    bond_indices = (
        torch.argwhere(
            trafo.transform.permute[:, None]
            == trafo.transform.ic_transform.bond_indices[None, :],
        )[:, 0]
        - 6
    )
    bond_indices = torch.concatenate([bond_indices, torch.IntTensor([0, 1])])

    angle_indices = (
        torch.argwhere(
            trafo.transform.permute[:, None]
            == trafo.transform.ic_transform.angle_indices[None, :],
        )[:, 0]
        - 6
    )
    angle_indices = torch.concatenate([angle_indices, torch.IntTensor([2])])

    dih_indices = (
        torch.argwhere(
            trafo.transform.permute[:, None]
            == trafo.transform.ic_transform.dih_indices[None, :],
        )[:, 0]
        - 6
    )

    return (
        torch.tensor(means_final).to(torch.get_default_dtype()),
        torch.tensor(stds_final).to(torch.get_default_dtype()),
        bond_indices,
        angle_indices,
        dih_indices,
    )


def test_unscale_internal_coordinates(all_atom_config):
    trafo = create_coord_trafo()

    all_atom_config_transformed = trafo.transform.forward(all_atom_config)[0]
    unscaled_all_atom_config_transformed = unscale_internal_coordinates(
        all_atom_config_transformed, trafo
    )

    all_atom_config = all_atom_config.numpy().reshape(-1, 22, 3)[0, :, :]
    all_atom_config_transformed = all_atom_config_transformed.numpy()
    unscaled_all_atom_config_transformed = unscaled_all_atom_config_transformed.numpy()

    def _calculate_distance(coord1, coord2):
        return np.linalg.norm(coord1 - coord2)

    def _calculate_angle(coord1, coord2, coord3):
        return np.arccos(
            np.dot(coord1 - coord2, coord3 - coord2)
            / (np.linalg.norm(coord1 - coord2) * np.linalg.norm(coord3 - coord2))
        )

    def _calculate_dihedral(coord1, coord2, coord3, coord4):
        coords = np.array([coord1, coord2, coord3, coord4])
        traj_from_npy = mdtraj.Trajectory(coords, None)

        return -1.0 * mdtraj_dihedral(  # fab uses a different sign convention
            traj_from_npy, np.array([0, 1, 2, 3])[None, :], periodic=False
        )

    output_info = get_trafo_output_info(trafo)

    unscaled_internal_coords = np.zeros(60)
    for i, item in enumerate(output_info):
        if item[0] == "bond":
            unscaled_internal_coords[i] = _calculate_distance(
                all_atom_config[item[1][0], :], all_atom_config[item[1][1], :]
            )
            # print(unscaled_internal_coords[i] - unscaled_min_config_transformed[0, i])
        elif item[0] == "angle":
            unscaled_internal_coords[i] = _calculate_angle(
                all_atom_config[item[1][0], :],
                all_atom_config[item[1][1], :],
                all_atom_config[item[1][2], :],
            )
            # print(unscaled_internal_coords[i] - unscaled_min_config_transformed[0, i])
        elif item[0] == "dihedral":
            unscaled_internal_coords[i] = _calculate_dihedral(
                all_atom_config[item[1][0], :],
                all_atom_config[item[1][1], :],
                all_atom_config[item[1][2], :],
                all_atom_config[item[1][3], :],
            )
            # Wrap it into [-pi, pi]
            unscaled_internal_coords[i] = (unscaled_internal_coords[i] + np.pi) % (
                2 * np.pi
            ) - np.pi

            # print(unscaled_internal_coords[i] - unscaled_min_config_transformed[0, i])

    # print("Difference:")
    diff = unscaled_all_atom_config_transformed - unscaled_internal_coords
    # print(diff)
    # print("Max difference:")
    max_diff = np.max(np.abs(diff))

    return max_diff


def test_trafo():
    trafo = create_coord_trafo()

    info = get_trafo_output_info(trafo)

    print(info)

    minimum_energy_config = get_minimum_energy_configuration()

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("phi, psi, theta_1, theta_2")
    print("Beginning:")
    print(all_dihedrals)

    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Figure out what this is in the internal coordinates (after removing the scaling and meaning):
    phi_scaled = minimum_energy_config[0, 17]
    index_in_dih = trafo.transform.permute[17 + 6]
    index_in_dih = torch.argwhere(
        trafo.transform.ic_transform.dih_indices == index_in_dih
    )[0, 0]
    phi_unscaled = (
        phi_scaled * trafo.transform.ic_transform.std_dih[index_in_dih]
        + trafo.transform.ic_transform.mean_dih[index_in_dih]
    )
    print(phi_unscaled)
    # => This shows that before scaling, the internal coordinates have a different sign convention

    # Change phi:
    minimum_energy_config[0, 17] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing phi:")
    print(all_dihedrals)

    # Transform again:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Change psi:
    minimum_energy_config[0, 44] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing psi:")
    print(all_dihedrals)

    # Transform again:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Change theta_1:
    minimum_energy_config[0, 8] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing theta_1:")
    print(all_dihedrals)

    # Transform again:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Change theta_2:
    minimum_energy_config[0, 50] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing theta_2:")
    print(all_dihedrals)

    atom_8 = minimum_energy_config.reshape(1, 22, 3)[0, 8, :]
    atom_6 = minimum_energy_config.reshape(1, 22, 3)[0, 6, :]

    print("Distance between atom 8 and 6:")
    print(torch.norm(atom_8 - atom_6))

    # Transform again:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Change angle of bond [8,6]
    minimum_energy_config[0, 0] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing distance of bond [8,6]:")
    print(all_dihedrals)

    # Calculate the distance between atom 8 and 6:
    atom_8 = minimum_energy_config.reshape(1, 22, 3)[0, 8, :]
    atom_6 = minimum_energy_config.reshape(1, 22, 3)[0, 6, :]

    print("Distance between atom 8 and 6:")
    print(torch.norm(atom_8 - atom_6))

    atom_1 = minimum_energy_config.reshape(1, 22, 3)[0, 1, :]
    atom_0 = minimum_energy_config.reshape(1, 22, 3)[0, 0, :]

    print("Distance between atom 1 and 0:")
    print(torch.norm(atom_1 - atom_0))

    # Transform again:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # Change angle of bond [8,6]
    minimum_energy_config[0, 3] = 12.0

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    all_dihedrals = _get_all_dihedrals(minimum_energy_config.numpy().reshape(1, 22, 3))[
        0, :
    ]
    print("After changing distance of bond [1,0]:")
    print(all_dihedrals)

    # Calculate the distance between atom 8 and 6:
    atom_1 = minimum_energy_config.reshape(1, 22, 3)[0, 1, :]
    atom_0 = minimum_energy_config.reshape(1, 22, 3)[0, 0, :]

    print("Distance between atom 1 and 0:")
    print(torch.norm(atom_1 - atom_0))

    print("\n\n")

    ### Check angle calculation is correct:

    # Determine angle between atoms 6,8,14 (index 2 in internal coordinates)
    atom_6 = minimum_energy_config.reshape(1, 22, 3)[0, 6, :]
    atom_8 = minimum_energy_config.reshape(1, 22, 3)[0, 8, :]
    atom_14 = minimum_energy_config.reshape(1, 22, 3)[0, 14, :]
    print("Angle between atoms 6,8,14:")
    print(
        torch.acos(
            torch.dot(atom_6 - atom_8, atom_14 - atom_8)
            / (torch.norm(atom_6 - atom_8) * torch.norm(atom_14 - atom_8))
        )
    )

    # Transform:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)
    print(
        minimum_energy_config[0, 2] * trafo.transform.std_angle
        + trafo.transform.mean_angle
    )

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)

    # Determine angle between atoms 4,1,0 (index 4 in internal coordinates)
    atom_4 = minimum_energy_config.reshape(1, 22, 3)[0, 4, :]
    atom_1 = minimum_energy_config.reshape(1, 22, 3)[0, 1, :]
    atom_0 = minimum_energy_config.reshape(1, 22, 3)[0, 0, :]

    print("Angle between atoms 4,1,0:")
    print(
        torch.acos(
            torch.dot(atom_4 - atom_1, atom_0 - atom_1)
            / (torch.norm(atom_4 - atom_1) * torch.norm(atom_0 - atom_1))
        )
    )

    # Transform:
    minimum_energy_config, _ = trafo.transform.forward(minimum_energy_config)

    # After permutation:
    index_in_angles = trafo.transform.permute[4 + 6]
    index_in_angles = torch.argwhere(
        trafo.transform.ic_transform.angle_indices == index_in_angles
    )[0, 0]
    print(
        minimum_energy_config[0, 4]
        * trafo.transform.ic_transform.std_angles[index_in_angles]
        + trafo.transform.ic_transform.mean_angles[index_in_angles]
    )

    # Transform back:
    minimum_energy_config, _ = trafo.transform.inverse(minimum_energy_config)


def get_dih_mean_scale(index, trafo):
    # After permutation:
    index_in_dih = trafo.transform.permute[index + 6]
    index_in_dih = torch.argwhere(
        trafo.transform.ic_transform.dih_indices == index_in_dih
    )[0, 0]

    return (
        trafo.transform.ic_transform.mean_dih[index_in_dih],
        trafo.transform.ic_transform.std_dih[index_in_dih],
    )


def alternative_filter_chirality(
    x_int: torch.Tensor,
    trafo: bg.flows.CoordinateTransform,
    output_float=False,
    supplied_x_cart=False,
    use_hydrogen_carbon_vector=False,
):
    """
    Calculate the chirality of alanine dipeptide.

    Parameters:
    x_int (torch.Tensor): Internal coordinates of the alanine dipeptide.
    trafo (bg.flows.CoordinateTransform): Coordinate transformation object.
    output_float (bool, optional): If true, additionally output the float. Defaults to False.
    supplied_x_cart (bool, optional): If true, x_int is assumed to be cartesian coordinates. Defaults to False.
    use_hydrogen_carbon_vector (bool, optional): If true, use the C_alpha - hydrogen vector instead of the
        nitrogen - hydrogen vector. Defaults to False.

    Returns:
    L_mask (torch.Tensor): Tensor of shape (x_int.shape[0],) containing the chirality of each configuration
        (True for L, False for D)
    """

    if not supplied_x_cart:
        x_cartesian = trafo.transform.inverse(x_int)[0].reshape(-1, 22, 3)
    else:
        x_cartesian = x_int.reshape(-1, 22, 3)

    center = x_cartesian[:, 8, :]  # alpha carbon
    a1 = x_cartesian[:, 6, :]  # nitrogen
    a2 = x_cartesian[:, 14, :]  # carbon
    a3 = x_cartesian[:, 10, :]  # carbon
    a4 = x_cartesian[:, 9, :]  # hydrogen

    # Vectors from center to each atom
    v1 = a1 - center
    v2 = a2 - center
    v3 = a3 - center
    v4 = a4 - center

    # Calculate the normal of the plane formed by v1, v2, v3
    normal = torch.cross(v2 - v1, v3 - v1)

    if not use_hydrogen_carbon_vector:
        # Determine the direction of the fourth group relative to the plane
        # direction = np.dot(normal, v4 - v1)
        direction = torch.sum(normal * (v4 - v1), dim=1)
    else:
        direction = torch.sum(normal * (a4 - center), dim=1)

    if output_float:
        return direction < 0, direction
    else:
        return direction < 0  # L_mask