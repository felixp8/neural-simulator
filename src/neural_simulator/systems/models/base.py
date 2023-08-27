import numpy as np
from typing import Optional, Union, Callable, Any


class Model:
    """Base class for all dynamics models"""

    def __init__(self, n_dim: int, seed=None):
        super().__init__()
        self.n_dim = n_dim
        self.rng = np.random.default_rng(seed)
    
    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_ics(
        self,
        ics: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        dist: Optional[Union[str, Callable]] = None,
        dist_params: dict = {},
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Sample initial conditions for dynamics model trajectories

        Parameters
        ----------
        ics : np.ndarray, optional
            If provided, overrides the sampling operation and just 
            returns these fixed ICs
        n : int, optional
            Number of samples to generate
        dist : str or callable, optional
            Either the name of a numpy random distribution or
            a function that generates random samples. If
            it is a function, it must accept a `size` argument
            specifying the shape of the sampled array
        dist_params : dict, optional
            Parameters for the generating samples from `dist`
        
        Returns
        -------
        ics : np.ndarray
            Array of shape (B,N) where B is batch size/trial count
            and N is dimensionality of the dynamics model
        """
        if ics is not None:
            return ics
        assert n is not None, "If `ics = None`, `n` must be provided"
        assert dist is not None, "If `ics = None`, `dist` must be provided"
        if dist == "zeros":
            return np.zeros((n, self.n_dim))
        elif hasattr(self.rng, dist):
            ics = getattr(self.rng, dist)(size=(n, self.n_dim), **dist_params)
        elif callable(dist):
            ics = dist(size=(n, self.n_dim), **dist_params)
        else:
            raise ValueError
        return ics
    
    def simulate(
        self, 
        ics: Optional[np.ndarray] = None, 
        inputs: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, Union[np.ndarray, None], Union[Any, None]]:
        """Simulate the dynamics model forward in time,
        given initial conditions and potentially inputs

        Parameters
        ----------
        ics : np.ndarray
            Initial conditions for dynamics model, with
            shape (B,N) where B is batch size/trial count
            and N is dimensionality of the dynamics model
        inputs : np.ndarray, optional
            Inputs to dynamics model at each timestep,
            with shape (B,T,I) where B is batch size/trial
            count, T is number of timesteps, and I is
            input dimensionality
        
        Returns
        -------
        trajectories : np.ndarray
            Sampled trajectories, with shape (B,T,N)
        outputs : np.ndarray, optional
            Corresponding model outputs, with shape
            (B,T,O), where O is output dimensionality
        action : any, optional
            System actions on the environment, can
            be any format expected by associated environment
        """
        raise NotImplementedError
    
