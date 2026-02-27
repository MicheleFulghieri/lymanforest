import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from scipy.interpolate import interp1d
from numba import njit, prange

# pyhisical constants
m_p     = 1.6726 * 1e-24     # proton mass in g
k_b     = 1.3806 * 1e-16     # boltzmann constant in erg/K
gamma   = 5/3                # adiabatic index (ideal gas)
c       = 2.9979 * 1e10      # speed of light in cm/s
sigma_0 = 4.45   * 1e-18     # Ly-alpha cross-section
