import numpy as np
from typing import Optional, Union

from .base import DataSampler


def temp_code():
    # args
    mode = "demean"
    T = None

    # "global"
    N = None  # neuron count
    u = None  # factor to neuron mapping
    u_in = None  # input to neuron mapping
    muf = -0.3  # mean connection strength
    mus = 0.0  # mean slow connection strength
    gs = None  # variance scaling for slow
    gf = None  # variance scaling for fast
    dt = None  # timestep

    # hard-coded vars
    Tinit = 50  # ?
    DTRLS = 2  # steps per RLS update
    tauV = 1e-2  # time constant for voltage
    taus = 1e-1  # time constant for slow spike input
    tauf = 5e-3  # time constant for fast spike input
    vth = 1e-16  # spike threshold
    vr = -10  # ?
    etaus = np.exp(-dt / taus)  # decay per dt for slow
    etauf = np.exp(-dt / tauf)  # decay per dt for fast
    etauV = np.exp(-dt / tauV)  # decay per dt for voltage

    # J_0 decomposed into mean and sampled variance
    Jmuf = muf * (1 / tauf) * 1 / (N) * np.ones(N, N)  # ?
    Jmus = mus * (1 / taus) * 1 / (N) * np.ones(N, N)
    J0f = gf * (1 / tauf) * 1 / np.sqrt(N) * np.random.randn(N, N)  # ?
    J0s = gs * (1 / taus) * 1 / np.sqrt(N) * np.random.randn(N, N)  # ?

    # what would be actual data
    factors = None
    targets = None
    m = output_size = targets.shape[-1]  # ?
    P = factor_size = factors.shape[-1]  # ?

    vJsbar = None

    if mode == "demean":
        sum_vJsbar = np.zeros(N, 1)
        vJsbar = np.zeros(N, 1)

    elif mode in ["train", "test", "data"]:
        nMSE = np.full((T, 1), np.nan)

        if mode == "train":
            W = np.zeros(m, P + 1)  # for computing learned output matrix
            PW = np.eye(P + 1)  # inverse covariance matrix

            w = np.zeros(P, 2 * N)  # P readout weights, half of J
            Pw = np.eye(2 * N)  # inverse covariance matrix

    # set up common vars
    y = np.zeros(P, 1)  # learned input current
    ss = np.zeros(N, 1)  # slow presynaptic current/firing rate when using RLS
    sf = np.zeros(N, 1)  # fast presynaptic current when using RLS

    v = 1e-3 * np.random.randn(N, 1)  # spike net state
    J0ss = np.zeros(N, 1)  # J0 slow input current
    J0fs = np.zeros(N, 1)  # J0 fast input current
    J0fbars = np.zeros(N, 1)  # J0 mean current

    # set various counters
    ttrial = np.inf  # time in current trial
    TTrial = 0  # total time in current trial
    t_trial_num = 0  # trial number
    ttime = 1  # time across all trials
    go = True  # flag to quit loop


class FactorBasedSpikingNetwork(DataSampler):
    def __init__(
        self,
        N=None,  # neuron count
        rank=None,
        input_dim=None,
        output_dim=None,
        mu_f=-0.3,  # mean connection strength
        mu_s=0.0,  # mean slow connection strength
        g=None,  # general connectivity strength
        g_s=None,  # variance scaling for slow
        g_f=None,  # variance scaling for fast
        dt=None,  # timestep
        tauV=1e-2,  # time constant for voltage
        taus=1e-1,  # time constant for slow spike input
        tauf=5e-3,  # time constant for fast spike input
        v_thresh=1e-16,  # spike threshold
        v_reset=-10,  # ?
        demean_steps=1,
        steps_per_update=2,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.u_fac = self.rng.uniform(low=-1.0, high=1.0, size=(N, rank)) * g
        self.u_in = self.rng.uniform(low=-1.0, high=1.0, size=(N, input_dim)) * g
        self.w_fac = np.zeros((rank, N * 2))
        self.w_out = np.zeros((output_dim, N * 2))

        self.v_thresh = v_thresh
        self.v_reset = v_reset

        self.dt = dt
        self.e_tau_s = np.exp(-dt / taus)  # decay per dt for slow
        self.e_tau_f = np.exp(-dt / tauf)  # decay per dt for fast
        self.e_tau_V = np.exp(-dt / tauV)  # decay per dt for voltage

        self.input_dim = input_dim
        self.rank = rank
        self.output_dim = output_dim
        self.N = N
        self.mu_f = mu_f
        self.mu_s = mu_s
        self.g_f = g_f
        self.g_s = g_s

        self.steps_per_update = steps_per_update

        self.J0_mu_f = mu_f * (1 / tauf) * 1 / (N) * np.ones(N, N)
        self.J0_mu_s = mu_s * (1 / taus) * 1 / (N) * np.ones(N, N)
        self.J0_std_f = g_f * (1 / tauf) * 1 / np.sqrt(N) * self.rng.normal(size=(N, N))
        self.J0_std_s = g_s * (1 / taus) * 1 / np.sqrt(N) * self.rng.normal(size=(N, N))

    def sample(
        self,
        trajectories: np.ndarray,
        inputs: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
    ):
        # demean
        demean_trials = 100
        vJsbar = np.zeros(self.N, 1)

        # set up common vars
        y = np.zeros(self.rank, 1)  # learned input current
        ss = np.zeros(self.N, 1)  # slow presynaptic current/firing rate when using RLS
        sf = np.zeros(self.N, 1)  # fast presynaptic current when using RLS

        v = 1e-3 * self.rng.normal(size=(self.N, 1))  # spike net state
        J0ss = np.zeros(self.N, 1)  # J0 slow input current
        J0fs = np.zeros(self.N, 1)  # J0 fast input current
        J0sbars = np.zeros(self.N, 1)  # J0 mean current
        J0fbars = np.zeros(self.N, 1)  # J0 mean current

        # set various counters
        ttrial = np.inf  # time in current trial
        TTrial = 0  # total time in current trial
        t_trial_num = -1  # trial number
        ttime = 0  # time across all trials
        go = True  # flag to quit loop

        total_steps = 0
        for iter in range(demean_trials):
            trial_num = self.rng.choice(trajectories.shape[0])
            factors = trajectories[trial_num]
            inputs = (
                np.zeros((factors.shape[0], 0)) if inputs is None else inputs[trial_num]
            )
            for t in range(factors.shape[0]):  # TODO: support jagged masked arrays
                y = factors[t]
                vinf = (
                    J0fbars
                    + J0sbars
                    - vJsbar
                    + self.u_fac * y
                    + J0ss
                    + J0fs
                    + self.u_in * inputs
                )
                v = vinf + (v - vinf) * self.e_tau_V  # Voltage

                S = v >= self.v_thresh  # spikes
                v[S] = self.v_thresh + self.v_reset  # reset

                J0ss = J0ss * self.e_tau_s + np.sum(
                    self.J0_std_s[:, S], -1
                )  # slow J0 currents
                J0fs = J0fs * self.e_tau_f + np.sum(
                    self.J0_std_f[:, S], -1
                )  # fast J0 currents
                J0sbars = J0sbars * self.e_tau_s + np.sum(
                    self.J0_mu_s[:, S], -1
                )  # mean slow J0 currents
                J0fbars = J0fbars * self.e_tau_f + np.sum(
                    self.J0_mu_f[:, S], -1
                )  # mean fast J0 currents

                # make presynaptic currents and concatenate
                ss = self.e_tau_s * ss + S
                sf = self.e_tau_f * sf + S

                # update mean current
                total_steps += 1
                vJsbar = vJsbar * (total_steps - 1 / total_steps) + (
                    self.u_fac * y + J0ss + J0fs
                ) * (1 / total_steps)

        # integrate factor approximating model
        vinf = J0fbars - vJsbar + self.u_fac * self.y + J0ss + J0fs + self.u_in * inputs
        v = vinf + (v - vinf) * self.e_tau_V  # Voltage

        S = v >= self.v_thresh  # spikes
        v[S] = self.v_thresh + self.v_reset  # reset

        # J0ss = J0ss * self.e_tau_s + sum(J0s(:,S),2) # slow J0 currents
        # J0fs = J0fs * self.e_tau_f + sum(J0f(:,S),2) # fast J0 currents
        # J0fbars = J0fbars * self.e_tau_f + sum(Jmu(:,S),2) # mean J0 currents

        # # make presynaptic currents and concatenate
        # ss = etaus * ss + S
        # sf = etauf * sf + S
        # s = [ss;sf]
