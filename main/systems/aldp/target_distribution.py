""" Code partially adapted from boltzgen repository: https://github.com/VincentStimper/boltzmann-generators
"""

from main.systems.aldp.simulation.initialize_simulation import get_system_aldp
import wandb
import os
from torch import nn
import torch
import multiprocessing as mp
import boltzgen.openmm_interface as omi


class TransformedBoltzmannParallel(nn.Module):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces and processes the batch of
    states in parallel
    """

    def __init__(
        self,
        system,
        temperature,
        do_apply_energy_regularization,
        energy_cut,
        energy_max,
        transform,
        n_threads=None,
    ):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param do_apply_energy_regularization: Whether to apply energy
        regularization (based on energy_cut and energy_max)
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        super().__init__()
        # Save input parameters
        self.system = system
        self.temperature = temperature

        if do_apply_energy_regularization:
            self.energy_cut = torch.tensor(energy_cut)
            self.energy_max = torch.tensor(energy_max)
        else:
            self.energy_cut = None
            self.energy_max = None

        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(
            self.n_threads,
            omi.OpenMMEnergyInterfaceParallel.var_init,
            (system, temperature),
        )

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterfaceParallel.apply

        if do_apply_energy_regularization:
            self.regularize_energy = omi.regularize_energy
        else:
            self.regularize_energy = lambda energy, energy_cut, energy_max: energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0], self.energy_cut, self.energy_max
        )

        self.transform = transform

    def log_prob(self, z):
        z_, log_det = self.transform(z)
        return -self.norm_energy(z_) + log_det


class _system:
    def __init__(self, topology, openmm_system):
        self.topology = topology
        self.system = openmm_system


def prepare_target(trafo, n_threads=None):
    openmm_system, prmtop = get_system_aldp(constraints=None, return_prmtop=True)

    system = _system(prmtop.topology, openmm_system)

    if n_threads is None:
        jobid = os.getenv("SLURM_JOB_ID")
        n_threads = (
            wandb.config.n_threads_cluster
            if (jobid is not None and jobid != "")
            else wandb.config.n_threads_local
        )

    target = TransformedBoltzmannParallel(
        system,
        temperature=wandb.config.temperature,
        do_apply_energy_regularization=wandb.config.get(
            "do_apply_energy_regularization", False
        ),
        energy_cut=wandb.config.energy_cut,
        energy_max=wandb.config.energy_max,
        transform=trafo,
        n_threads=n_threads,
    )

    return target


def calculate_log_prob_transformed(z, trafo, target=None):
    target_created_here = False
    if target is None:
        target = prepare_target(trafo)
        target_created_here = True

    # This outputs energies in units of kT
    log_prob = target.log_prob(z)

    if target_created_here:
        target.pool.terminate()
        target.pool.close()

    return log_prob
