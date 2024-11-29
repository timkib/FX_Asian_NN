import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from numba import njit, jit
from Dupire_general import dupire_local_volatility_surface
from scipy.interpolate import interp1d


def GBM(S0, r_dom, r_for, sigma, T, N, m):
    """Simulates N different geometric brownian motion pathes with m equidistant pathes."""
    r = r_dom - r_for  # fx adjustment
    sim_pathes = np.zeros(shape=(N, m + 1))
    sim_pathes[:, 0] = S0
    dt = T / (m)
    for i in range(N):
        brownian_motion = np.random.normal(loc=0.0, scale=1.0, size=m)
        sim_pathes[i, 1:] = S0 * np.cumprod(np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * brownian_motion))

    return sim_pathes


def GBM_QMC(number_paths, number_time_steps,
            S_0, r_dom, r_for, sigma, T):
    """Simulates n geometric brownian motion of length k using the closed-form solution and Sobol sequences.

    :number_paths:                  number of paths
    :number_time_steps:             number of time discretizations
    :r_dom:                         riskless domestic interest rate
    :r_for:                         riskless foreign interest rate
    :sigma:                         (constant) volatility
    :S_0:                           starting stock price
    :return:                        matrix with simulated stock prices
    """
    S = np.ones(shape=(number_paths, number_time_steps))
    r = r_dom - r_for
    # apply Quasi Monte-Carlo
    W = qmc.MultivariateNormalQMC(mean=np.zeros(shape=(number_time_steps)),
                                  cov=np.identity(number_time_steps)).random(number_paths)
    delta_t = T / (number_time_steps - 1)
    W[:, 0] = 0
    S[:, 0] = S_0

    S[:, 1:] = np.exp((r - sigma ** 2 / 2) * delta_t + sigma * W[:, 1:] * np.sqrt(delta_t))  # GBM

    return np.cumprod(S, axis=1)  # calculate GBM


def GBM_IV(S0, r_dom, r_for, sigma, T, N, m, numb_generator=None):
    """Simulates N different geometric brownian motion paths with m equidistant pathes in the Local Vol model."""
    r = r_dom - r_for  # fx adjustment
    sim_paths = np.zeros(shape=(N, m + 1))
    sim_paths[:, 0] = S0
    dt = T / (m)
    if numb_generator == "QMC":
        # apply Quasi Monte-Carlo
        brownian_motions = qmc.MultivariateNormalQMC(mean=np.zeros(shape=(m + 1)),
                                                     cov=np.identity(m + 1)).random(N)
    else:
        brownian_motions = np.random.normal(loc=0.0, scale=1.0, size=(N, m))
    for i in range(m):
        sim_paths[:, i + 1] = np.exp((r - sigma[i] ** 2 / 2) * dt + sigma[i] * np.sqrt(dt) * brownian_motions[:, i])

    sim_paths = np.cumprod(sim_paths, axis=1)

    return sim_paths


def GBM_LocalVol(S0, r_dom, r_for, T, times, local_volatilities,
                 numb_generator=None, m=300):
    """Simulates stock paths in the local vol model using Euler approximation scheme."""
    r = r_dom - r_for  # fx adjustment
    N = 2 ** 17
    m = int(365 * T)
    sim_paths = np.zeros(shape=(N, m + 1))
    sim_paths[:, 0] = S0

    sim_paths[:, 0] = S0
    dt = T / (m)
    if numb_generator == "QMC":
        W = qmc.MultivariateNormalQMC(mean=np.zeros(shape=(m + 1)),
                                      cov=np.identity(m + 1)).random(N)
    else:
        W = np.random.normal(loc=0.0, scale=1.0, size=(N, m))

    for i in range(m):
        t_i = dt * (i + 1)
        local_vol_val = dupire_local_volatility_surface(times, local_volatilities)(
            t_i)  # uses interpolated dupire volas
        d_St = r * sim_paths[:, i] * dt + np.sqrt(local_vol_val) * sim_paths[:, i] * np.sqrt(dt) * (W[:, i])
        sim_paths[:, i + 1] = sim_paths[:, i] + d_St  # GBM SDE

    return sim_paths


def GBM_LocalVol_Milstein(S0, r_dom, r_for, T, times, local_volatilities, numb_generator=None, m=300):
    """Simulates stock paths in the local vol model using Milstein approximation scheme."""
    r = r_dom - r_for  # fx adjustment
    N = 2 ** 17
    m = int(365 * T)
    sim_paths = np.zeros(shape=(N, m + 1))
    sim_paths[:, 0] = S0

    sim_paths[:, 0] = S0
    dt = T / (m)
    if numb_generator == "QMC":
        W = wrapper_QMC(N, m)
    else:
        W = wrapper_NpRandom(N, m)

    sim_paths = wrapper_sde(sim_paths, m, dt, times, local_volatilities, W, r)

    return sim_paths


def wrapper_QMC(N, m):
    return qmc.MultivariateNormalQMC(mean=np.zeros(shape=(m + 1)), cov=np.identity(m + 1)).random(N)


@njit
def wrapper_NpRandom(N, m):
    return np.random.normal(loc=0.0, scale=1.0, size=(N, m))


@njit
def wrapper_sde(sim_paths, m, dt, times, local_volatilities, W, r):
    for i in range(m):
        t_i = dt * (i + 1)
        # local_vol_val = dupire_local_volatility_surface(times, local_volatilities)(t_i)  # uses interpolated volas
        local_vol_val = np.interp(t_i, times, local_volatilities)  # has to be checked
        d_St = r * sim_paths[:, i] * dt + np.sqrt(local_vol_val) * sim_paths[:, i] * np.sqrt(dt) * (
            W[:, i]) + 0.5 * local_vol_val * sim_paths[:, i] * ((np.sqrt(dt) * (W[:, i])) ** 2 - dt)
        # print("d_St", d_St)
        sim_paths[:, i + 1] = sim_paths[:, i] + d_St  # GBM SDE

    return sim_paths

# TEST
# S0 = 1.2
# T = 1
# sigma = 0.9
# K = 1.2
# time_steps = 3
# r = 0.05
# r_dom = r
# r_for = r_dom
# fixings = np.array([0.5, 1.0])
# N = 2 ** 17  # - 1
