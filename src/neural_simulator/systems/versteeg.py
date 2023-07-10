from task_modeling.task_envs import TaskGenerator
import pandas as pd
import ivy

from .base import UncoupledSystem


class TaskGeneratorSystem(UncoupledSystem):
    """Dynamical system that takes input from TaskGenerator object"""

    def __init__(
        self,
        system,
        env: TaskGenerator,
        n_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_dim=n_dim, n_input_dim=len(env.input_labels), seed=seed)
        self.system = system
        self.env = env
        # self.env.seed(seed)

    def sample_trials(self, n_trials: int) -> list[dict]:
        # noop because handled internally by taskgenerator
        # BUT this also means we have no trial info so would need to make
        # changes to that
        return [{} for _ in range(n_trials)]

    def sample_inputs(
        self,
        trial_info: list[dict],
    ) -> np.ndarray:
        self.env.n_samples = len(trial_info)
        inputs, _ = self.env.generate_dataset()
        for i in range(inputs.shape[0]): # should modify the trial_info in place so don't need to return it???
            trial_info[i].update(dict(zip(arr[i, 0, :], self.env.input_labels)))
        # TODO (on this and gym envs): keep outputs, compute losses so we know bad and good trials
        return inputs

    def simulate_system(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # model forward pass
        raise NotImplementedError
