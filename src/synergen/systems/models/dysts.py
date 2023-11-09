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
        save_ics: bool = False,
    ) -> tuple[np.ndarray, Optional[np.ndarray], Any, Optional[dict]]:
        """Simulate states from dysts dynamical system

        Parameters
        ----------
        ics : np.ndarray
            Initial conditions for dynamics model, with
            shape (B,N) where B is batch size/trial count
            and N is dimensionality of the dynamics model
        trial_len : int
            Lengths of states to simulate
        burn_in : int, default: 0
            Number of samples to drop from the beginning of
            simulations as a "burn-in" period
        method : str, default: "Radau"
            Integration method. See `dysts`
        resample : bool, default: True
            Whether to resample states to have
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
        save_ics : bool, default: False
            Whether to save the sampled initial conditions
            in `batched_data` or not

        Returns
        -------
        states : np.ndarray
            Sampled states, with shape (B,T,N)
        outputs : None
            Does not return outputs
        actions : None
            Does not return outputs
        temporal_data : None
            Does not return additional temporal data
        """
        states = []
        for ic in ics:
            self.flow.ic = ic
            state_traj = self.flow.make_trajectory(
                n=(trial_len + burn_in),
                method=method,
                resample=resample,
                pts_per_period=pts_per_period,
                standardize=standardize,
                postprocess=postprocess,
                noise=noise,
            )[burn_in:]
            states.append(state_traj)
        states = np.stack(states, axis=0)
        return states, None, None, None
