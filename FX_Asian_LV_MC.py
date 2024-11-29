import numpy as np
from FX_ASIAN_MC import GBM_IV, GBM_LocalVol, GBM_LocalVol_Milstein
from helper import get_closest
from numba import njit, jit


@njit
def call_payout(stock_prices, strike):
    return np.maximum(stock_prices - strike, 0)


class AsianSmileFXBSMC:
    """Calculates the pv of an Asian option with arithmetic mean in the BS market with smile for FX options."""

    def __init__(self, S0, r_dom, r_for, sigma, T, K, fixings, notional, m, N=2 ** 16, type_paths="GBM"):
        self.S0 = S0
        self.r = r_dom - r_for  # fx adjustment
        self.r_dom = r_dom
        self.r_for = r_for
        self.sigma = sigma
        self.T = T
        self.K = K
        self.fixings = fixings
        self.discount_factor = np.exp(-self.r_dom * self.T)
        self.n = len(fixings)
        self.notional = notional
        self.N = N
        self.m = m
        self.type_paths = type_paths

    def mean_stocks(self, stock_paths):
        """Calculates the mean of the given simulated stock prices."""
        return np.mean(stock_paths[:, 1:], axis=1)

    def pv(self):
        if self.type_paths == "GBM":
            stock_paths = GBM_IV(self.S0, self.r_dom, self.r_for, self.sigma, self.T, self.N, self.m)
        elif self.type_paths == "QMC":
            stock_paths = GBM_IV(self.S0, self.r_dom, self.r_for, self.sigma, self.T, self.N, self.m, "QMC")
        else:
            stock_paths = 0
        mean_stock_prices = self.mean_stocks(stock_paths)
        # MC estimator
        V0 = self.discount_factor * np.mean(call_payout(mean_stock_prices, self.K))  # MC
        return V0 * self.notional


class AsianLocalVolMC:
    """Calculates the pv of an Asian option with arithmetic mean in the Local Vol market for FX options."""

    def __init__(self, S0, r_dom, r_for, T, K, fixings, notional,
                 times, local_volatilities,
                 N=2 ** 17, m=365, type_paths="GBM", scheme="Euler"):
        self.S0 = S0
        self.r = r_dom - r_for  # fx adjustment
        self.r_dom = r_dom
        self.r_for = r_for
        self.T = int(T)
        self.K = K
        self.fixings = np.unique(fixings)
        self.discount_factor = np.exp(-self.r_dom * self.T)
        self.n = len(fixings)
        self.notional = notional
        self.N = N
        self.m = m
        self.type_paths = type_paths
        self.times = times # local vol time vector
        self.local_volatilities = local_volatilities
        self.scheme = scheme
        self.dt = 1 / 365
        self.numb_fixings = len(self.fixings)

    def mean_stocks(self, stock_paths):
        """Calculates the mean of the given simulated stock prices."""
        fixing_idx = np.array((self.fixings * 365) - 1, dtype=int)
        return np.mean(stock_paths[:, fixing_idx], axis=1)

    def pv(self):
        if self.scheme == "Euler":
            stock_paths = GBM_LocalVol(self.S0, self.r_dom, self.r_for, self.T, self.times, self.local_volatilities,
                                       m=self.m, numb_generator=self.type_paths)
        elif self.scheme == "Milstein":
            stock_paths = GBM_LocalVol_Milstein(self.S0, self.r_dom, self.r_for, self.T, self.times,
                                                self.local_volatilities, m=self.m,
                                                numb_generator=self.type_paths)
        else:
            stock_paths = 0
        mean_stock_prices = self.mean_stocks(stock_paths)
        # MC estimator
        V0 = self.discount_factor * np.mean(call_payout(mean_stock_prices, self.K))  # MC
        return V0 * self.notional
