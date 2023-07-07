import abc

class NeuralSimulator(abc.ABC):
    def __init__(self, system, env=None):
        super().__init__()
        self.system = system
        self.env = env

    def generate_dataset(self, **kwargs):
        trajectories = self.sample_trajectories(**kwargs)
        neural_data = self.simulate_neural_data(trajectories, **kwargs)

    @abstractmethod
    def sample_trajectories(self, **kwargs):
        pass

    @abstractmethod
    def simulate_neural_data(trajectories, **kwargs):
        pass
