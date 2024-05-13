from main.active_learning.free_energy_ensemble import FreeEnergyEnsemble


class ActiveLearningSampler:
    def __init__(self, ensemble: FreeEnergyEnsemble):
        self.ensemble = ensemble

    def sample_new_higherror_points(self):
        raise NotImplementedError("Subclass must implement this method")
