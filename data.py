# Generate dataset
# setting: 2 fixing times, T=1, K is fixed

import numpy as np
from Dupire_FX import sigma_tilde, d_sigma_dt, DupireFormulaFX_1D
from FX_Asian_LV_MC import AsianLocalVolMC
import time
from numba import njit
from scipy.stats import qmc
import numba.cuda


class DataGeneratorFixed:
    def __init__(self, length, vola_times, implied_volatilities):
        self.S0 = 1.2  # fixed
        self.T = 3.0  # fixed
        self.K = 1.2  # fixed
        self.r_dom = 0.05  # fixed
        self.r_for = 0.05  # fixed
        self.notional = 100  # fixed
        self.delta = self.T / 365  # one year
        self.length = length  # Number of data points
        self.vola_times = vola_times
        self.implied_volatilities = implied_volatilities
        self.numb_features = 5  # time_val1, time_val2, implied_vol1, implied_vol2, PV_MC
        self.numb_rows = self.length ** 2
        self.DATA = np.zeros(shape=(self.numb_rows, self.numb_features))  # initialize data matrix
        self.fill_datamatrix()
        self.remove_duplicates()
        self.add_implied_vol()
        self.add_MC()

    def create_meshgrid(self):
        """Creates a mesh grid for t_1 and t_2 values."""
        x = np.linspace(self.delta, self.T, self.length)
        y = np.linspace(self.delta, self.T, self.length)
        return np.meshgrid(x, y)

    def fill_datamatrix(self):
        """Adds the mesh grid to the existing data matrix."""
        xv, yv = self.create_meshgrid()
        self.DATA[:self.numb_rows, 0] = yv.ravel()  # flatten array
        self.DATA[:self.numb_rows, 1] = xv.ravel()  # flatten array

    def remove_duplicates(self):
        """Removes rows where t_1 >= t_2, i.e. date of fixing one is after date of fixing date 2."""
        idx_smaller = self.DATA[:, 0] < self.DATA[:,
                                        1]  # ids where column time value 1 is smaller than column time val 2
        self.DATA = self.DATA[idx_smaller]

    def add_implied_vol(self):
        """Given the data matrix time points adds the respective implied sigma."""
        for row in range(self.DATA.shape[0]):
            for column in range(2):
                time_value = self.DATA[row, column]
                if time_value <= self.vola_times[0]:  # set extrapolation value
                    sigma_value = self.implied_volatilities[0]
                else:  # interpolation
                    sigma_value = sigma_tilde(self.vola_times, self.implied_volatilities, t=time_value)
                self.DATA[row, column + 2] = sigma_value

    def get_local_vols(self):
        """Return the local volatilities and the time points using the 1D Dupire formula."""
        local_vol_times = np.zeros(shape=367) # TODO
        local_vol_times[1:] = np.linspace(0.02, 1.1, 366) # TODO
        local_vols = [DupireFormulaFX_1D(self.K, self.implied_volatilities, self.vola_times, t, self.r_dom, self.r_for)
                      for t in local_vol_times[1:]]
        local_vols.insert(0, local_vols[0])
        local_vols = np.array(local_vols, dtype=np.float64)
        return local_vol_times, local_vols

    def add_MC(self):
        """Given the data matrix time points and implied volas adds the respective MC price."""
        local_vol_times, local_vols = self.get_local_vols()
        for i in range(self.DATA.shape[0]):
            fixings = self.DATA[i, : 2]
            asian_lv_mc = AsianLocalVolMC(self.S0, self.r_dom, self.r_for, self.T, self.K, fixings, self.notional,
                                          local_vol_times, local_vols,
                                          scheme="Milstein", type_paths="GBM").pv()
            self.DATA[i, -1] = asian_lv_mc  # asian_lv_mc  # assign MC value

    def get_data(self):
        """Returns the filled data matrix."""
        return self.DATA


class DataGeneratorFlexible(DataGeneratorFixed):
    def __init__(self, vola_times, implied_volatilities, maturities,
                 number_of_datapoints, max_numb_fixings):
        self.S0 = 1.2  # fixed
        self.T = int(maturities)
        self.K = 1.2  # fixed
        self.r_dom = 0.05  # fixed
        self.r_for = 0.05  # fixed
        self.notional = 100  # fixed
        self.length = 9 # int(self.T * 365)  # Number of data points
        self.delta = self.T / self.length  # first day
        self.number_of_datapoints = number_of_datapoints
        self.max_numb_fixings = int(max_numb_fixings)
        self.vola_times = vola_times
        self.implied_volatilities = implied_volatilities
        self.numb_features = 2 * max_numb_fixings + 1  # fixings + local volatilities + MC price
        self.numb_rows = int(self.length ** self.max_numb_fixings)
        self.DATA = np.zeros(shape=(self.numb_rows, self.numb_features))  # initialize data matrix
        self.fill_datamatrix()
        self.remove_duplicates()
        self.add_implied_vol()
        self.add_MC()

    def create_meshgrid(self):
        x = np.ones(shape=self.length) * self.delta
        x = np.cumsum(x)
        y = np.ones(shape=self.length) * self.delta
        y = np.cumsum(y)
        if self.max_numb_fixings == 2:
            return np.meshgrid(x, y)
        elif self.max_numb_fixings == 3:
            z = np.ones(shape=self.length) * self.delta
            z = np.cumsum(z)
            return np.meshgrid(x, y, z)
        else:
            raise Exception("No valid number of fixing numbers.")

    def remove_duplicates(self):
        """Removes rows where t_1 >= t_2, i.e. date of fixing one is after date of fixing date 2."""
        if self.max_numb_fixings == 2:
            idx_smaller = self.DATA[:, 0] <= self.DATA[:, 1]
        elif self.max_numb_fixings == 3:
            condition1 = np.argwhere(self.DATA[:, 0] <= self.DATA[:, 1])
            condition2 = np.argwhere(self.DATA[:, 1] <= self.DATA[:, 2])
            condition3 = np.argwhere(self.DATA[:, 0] <= self.DATA[:, 2])
            conditions_united1 = np.intersect1d(condition1, condition2)
            idx_smaller = np.intersect1d(conditions_united1, condition3)
        else:
            raise Exception("No valid number of fixing numbers.")
        self.DATA = self.DATA[idx_smaller]
        # random_ids = np.random.randint(0, self.DATA.shape[0], size=self.number_of_datapoints)
        # self.DATA = self.DATA[random_ids, :]

    def fill_datamatrix(self):
        """Adds the mesh grid to the existing data matrix."""
        if self.max_numb_fixings == 2:
            xv, yv = self.create_meshgrid()
            self.DATA[:self.numb_rows, 0] = yv.ravel()  # flatten array
            self.DATA[:self.numb_rows, 1] = xv.ravel()  # flatten array
        elif self.max_numb_fixings == 3:
            xv, yv, zv = self.create_meshgrid()
            self.DATA[:self.numb_rows, 0] = yv.ravel()  # flatten array
            self.DATA[:self.numb_rows, 1] = xv.ravel()  # flatten array
            self.DATA[:self.numb_rows, 2] = zv.ravel()  # flatten array
        else:
            raise Exception("No valid number of fixing numbers.")

    def add_implied_vol(self):
        """Given the data matrix time points adds the respective implied sigma."""
        for row in range(self.DATA.shape[0]):
            for column in range(self.max_numb_fixings):
                time_value = self.DATA[row, column]
                if time_value <= self.vola_times[0]:  # set extrapolation value
                    sigma_value = self.implied_volatilities[0]
                else:  # interpolation
                    sigma_value = sigma_tilde(self.vola_times, self.implied_volatilities, t=time_value)
                self.DATA[row, column + self.max_numb_fixings] = sigma_value

    def add_MC(self):
        """Given the data matrix time points and implied volas adds the respective MC price."""
        local_vol_times, local_vols = self.get_local_vols()
        for i in range(self.DATA.shape[0]):
            fixings = self.DATA[i, : self.max_numb_fixings]
            asian_lv_mc = AsianLocalVolMC(self.S0, self.r_dom, self.r_for, self.T, self.K, fixings, self.notional,
                                          local_vol_times, local_vols,
                                          scheme="Milstein", type_paths="QMC").pv()
            self.DATA[i, -1] = asian_lv_mc  # asian_lv_mc  # assign MC value

    def get_local_vols(self):
        """Return the local volatilities and the time points using the 1D Dupire formula."""
        local_vol_times = np.zeros(shape=int(self.T * 365 + 2))
        local_vol_times[1:] = np.linspace(0.02, self.T + 0.01, int(self.T * 365) + 1)

        local_vols = [DupireFormulaFX_1D(self.K, self.implied_volatilities, self.vola_times, t, self.r_dom, self.r_for)
                      for t in local_vol_times[1:]]
        local_vols.insert(0, local_vols[0])
        local_vols = np.array(local_vols, dtype=np.float32)
        return local_vol_times, local_vols


# implied_sigmas = np.array([0.38, 0.35, 0.31, 0.35, 0.45, 0.55, 0.1])
implied_sigmas = np.array([0.909, 0.915, 0.919, 0.927, 0.933, 0.938, 0.942])
one_week = 7 / 365
one_month = 30 / 365
two_months = one_month * 2
three_months = one_month * 3
six_months = one_month * 6
one_year = 1
two_years = 2

times = np.array([one_week, one_month, two_months, three_months, six_months, one_year, two_years])
start = time.time()
data_generator_flex = DataGeneratorFlexible(vola_times=times, implied_volatilities=implied_sigmas,
                                            maturities=1.0, number_of_datapoints=2, max_numb_fixings=2)
end = time.time()
data = data_generator_flex.get_data()
print("time: ", end - start)
np.set_printoptions(suppress=True)

print(data.shape)
print(data)
