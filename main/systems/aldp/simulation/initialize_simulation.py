from openmm.app import (
    AmberPrmtopFile,
    AmberInpcrdFile,
    Simulation,
    NoCutoff,
    OBC1,
)
from openmm import Platform, LangevinMiddleIntegrator
from openmm.unit import picoseconds, picosecond, kelvin
import os

# Code from https://github.com/choderalab/openmm-tutorials/tree/master
# See also https://github.com/choderalab/openmmtools/blob/main/openmmtools/testsystems.py


def get_system_aldp(constraints=None, return_prmtop=False):
    dirname = os.path.dirname(__file__)  # dirname of this file

    prmtop = AmberPrmtopFile(
        os.path.join(
            dirname, "input_files/aldp_implicit/alanine-dipeptide-implicit.prmtop"
        )
    )

    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        nonbondedCutoff=None,
        implicitSolvent=OBC1,
        constraints=constraints,
    )

    if return_prmtop:
        return system, prmtop
    else:
        return system


def initialize_simulation_aldp(constraints=None, platform: str = "CUDA") -> Simulation:
    platform = Platform.getPlatformByName(platform)

    dirname = os.path.dirname(__file__)  # dirname of this file

    inpcrd = AmberInpcrdFile(
        os.path.join(
            dirname, "./input_files/aldp_implicit/alanine-dipeptide-implicit.inpcrd"
        )
    )

    system, prmtop = get_system_aldp(constraints=constraints, return_prmtop=True)

    positions = inpcrd.positions
    topology = prmtop.topology

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.001 * picoseconds
    )
    simulation = Simulation(topology, system, integrator, platform=platform)
    simulation.context.setPositions(positions)

    return simulation
