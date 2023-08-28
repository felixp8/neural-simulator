import numpy as np
from dysts import flows
from typing import Optional, Union, Any
from .base import Model


class DystsModel(Model):
    """Dynamical systems implemented in `dysts`"""

    def __init__(
        self,
        name: str, 
        params: dict = {}, 
        seed: Optional[int] = None,
    ):
        self.flow = getattr(flows, name)(**params)
        self.flow.random_state = seed
        super().__init__(
            n_dim=self.flow.embedding_dimension,
            seed=seed,
        )
    
    def simulate(
        self,
        ics: np.ndarray,
        trial_len: int,
        burn_in: int = 0,
        method: str = "Radau",
        resample: bool = True,
        pts_per_period: int = 100,
        standardize: bool = False,
        postprocess: bool = True,
        noise: float = 0.0,
        inputs: Optional[Any] = None,
    ) -> tuple[np.ndarray, Union[np.ndarray, None], Union[Any, None]]:
        """Simulate trajectories from dysts dynamical system

        Parameters
        ----------
        ics : np.ndarray
            Initial conditions for dynamics model, with
            shape (B,N) where B is batch size/trial count
            and N is dimensionality of the dynamics model
        trial_len : int
            Lengths of trajectories to simulate
        burn_in : int, default: 0
            Number of samples to drop from the beginning of
            simulations as a "burn-in" period
        method : str, default: "Radau"
            Integration method. See `dysts`
        resample : bool, default: True
            Whether to resample trajectories to have 
            matching dominant Fourier components. See `dysts`
        pts_per_period : int, default: 100
            If resampling, the number of points per period.
            See `dysts`
        standardize: bool, default: False
            Whether to standardize the output time series.
            See `dysts`
        postprocess: bool, default: True
            Whether to apply coordinate conversions and other 
            domain-specific rescalings to the integration 
            coordinates. See `dysts`
        inputs : optional
            Completely ignored
        
        Returns
        -------
        trajectories : np.ndarray
            Sampled trajectories, with shape (B,T,N)
        outputs : NoneType
            Does not return outputs
        """
        trajectories = []
        for ic in ics:
            self.flow.ic = ic
            traj = self.flow.make_trajectory(
                n=(trial_len + burn_in),
                method=method,
                resample=resample,
                pts_per_period=pts_per_period,
                standardize=standardize,
                postprocess=postprocess,
                noise=noise,
            )[burn_in:]
            trajectories.append(traj)
        trajectories = np.stack(trajectories, axis=0)
        return trajectories, None, None