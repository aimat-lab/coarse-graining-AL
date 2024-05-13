from openmm.app import StateDataReporter
from sys import stdout
from reporters.force_reporter import ForceReporter
from reporters.position_reporter import PositionReporter
from datetime import datetime
from openmm.app.dcdreporter import DCDReporter
from initialize_simulation import initialize_simulation_aldp

if __name__ == "__main__":
    simulation = initialize_simulation_aldp(constraints=None, platform="CUDA")

    ########################################

    N_steps_starting = 50000
    N_ns_to_equilibrate = 50
    N_ns_to_simulate = 5000  # 5 microseconds

    write_traj_forces_interval = 250  # every 0.25 ps => results in 2e7 frames
    stdout_report_interval = 50000  # every 50 ps

    ########################################

    N_steps_equilibrate = 1000000 * N_ns_to_equilibrate
    N_steps_simulate = 1000000 * N_ns_to_simulate  # since we are using a 1 fs timestep

    print("\nMinimizing energy...\n")
    simulation.minimizeEnergy()

    curr_date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    def add_reporters(
        stdout_report_interval, write_traj_forces_interval, totalSteps, tag
    ):
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                stdout_report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                time=True,
                remainingTime=True,
                speed=True,
                progress=True,
                totalSteps=totalSteps,
                separator="\t\t",
            )
        )
        simulation.reporters.append(
            ForceReporter(
                f"./trajectories/{tag}_forces_{curr_date_time}.npy",
                write_traj_forces_interval,
                buffer_size=100,
            )
        )
        simulation.reporters.append(
            PositionReporter(
                f"./trajectories/{tag}_positions_{curr_date_time}.npy",
                write_traj_forces_interval,
                buffer_size=100,
            )
        )
        simulation.reporters.append(
            DCDReporter(
                f"./trajectories/{tag}_positions_{curr_date_time}.dcd",
                write_traj_forces_interval,
                enforcePeriodicBox=False,
            )
        )

    ### Starting ###
    add_reporters(
        stdout_report_interval=100,
        write_traj_forces_interval=1,
        totalSteps=int(N_steps_starting),
        tag="starting",
    )
    print("\nStarting steps...\n")
    simulation.step(int(N_steps_starting))
    del simulation.reporters[-4:]  # Remove the reporters

    add_reporters(
        stdout_report_interval=stdout_report_interval,
        write_traj_forces_interval=write_traj_forces_interval,
        totalSteps=int(N_steps_equilibrate),
        tag="eq",
    )
    print("\nEquilibrating...\n")
    simulation.step(int(N_steps_equilibrate))
    del simulation.reporters[-4:]  # Remove the reporters

    add_reporters(
        stdout_report_interval=stdout_report_interval,
        write_traj_forces_interval=write_traj_forces_interval,
        totalSteps=int(N_steps_simulate),
        tag="prod",
    )
    simulation.currentStep = 0
    print("\nSimulating production...\n")
    simulation.step(int(N_steps_simulate))

    print("\nDone!\n")
