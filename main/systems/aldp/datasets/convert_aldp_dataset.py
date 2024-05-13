import os
import torch
from main.systems.aldp.coord_trafo import create_coord_trafo
import numpy as np


def convert_fab_dataset(filename):
    file_path = os.path.join("./", filename)
    output_file_path = os.path.join("./", filename.split(".")[0] + "_cartesian.npy")

    dataset = torch.load(file_path)

    trafo = create_coord_trafo(
        minimum_energy_configuration_path="./position_min_energy_fab.pt"
    )

    # Convert dataset using batches of 1e6:
    cartesian = []
    for i in range(0, dataset.shape[0], int(1e6)):
        cartesian.append(trafo.transform.inverse(dataset[i : i + int(1e6)])[0])
    cartesian = torch.cat(cartesian, dim=0)

    print(f"Dataset {filename} has shape {dataset.shape} before conversion.")

    # nm to angstrom (OpenMM also uses angstrom, we need the same format here)
    cartesian *= 10.0
    cartesian = cartesian.reshape(-1, 22, 3)

    print(f"Dataset {filename} has shape {cartesian.shape} after conversion.")

    np.save(output_file_path, cartesian.numpy())


if __name__ == "__main__":
    convert_fab_dataset("test.pt")
