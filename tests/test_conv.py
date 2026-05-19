from src.python_zapocet_funkce import get_dataset, process_dataset, get_plasma_times, fit_profile_gauss_regression
import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

shotlist = [
    51820, 51821, 51822, 51823, 51824, 51825, 51826, 51827, 51828, 51829, 51830, 51831, 51832, 51833, 51834
]
rs = [98, 94, 90, 86, 82, 78, 74, 70, 66, 62, 58, 54, 50, 46, 42]

alpha = 2.0
BPP_alpha = 0.9
Te_shift = 0.5

R_LP = 100
R_BPP = 100

expected_size = 125000

def test_dataset():
    ds = get_dataset(shotlist, rs, alpha, Te_shift, BPP_alpha, R_BPP, R_LP)
    assert ds.t.size==expected_size, f"Expected dataset size {expected_size}, but got {ds.t.size}"


step_phi = 2

C_min_phi=1e-2
C_max_phi=1e3
RBF_min_phi=2e1
RBF_max_phi=2e2

noise_min = 1e-4
noise_max = 1e-1

n_samples = 20

def test_convergence():
    t1, t2 = get_plasma_times(shotlist)
    ds_mean_phi, ds_std_phi, t_bins_phi = process_dataset(
    shotlist, rs, alpha, Te_shift, BPP_alpha, R_BPP, R_LP, t1, t2, step_phi
    )
    to_cycle_phi = ds_mean_phi.t_bins.size

    for index in range(to_cycle_phi):
        da_phi, da_phi_std, realizations = fit_profile_gauss_regression(
                ds_mean_phi["phi"].isel(t_bins=index),
                ds_std_phi["phi"].isel(t_bins=index),
                C_min_phi,
                C_max_phi,
                RBF_min_phi,
                RBF_max_phi,
                noise_min,
                noise_max,
                n_samples,
            ),
        
        phi_discrete_r = ds_mean_phi["phi"].isel(t_bins=index).r.data
        phi_discrete = ds_mean_phi["phi"].isel(t_bins=index).data

        da_phi_interp = da_phi.interp(r=phi_discrete_r)
        da_phi_interp_std = da_phi_std.interp(r=phi_discrete_r)

        diff=np.abs(da_phi_interp.data - phi_discrete)

        assert np.all(diff < 4*da_phi_interp_std), f"Convergence test failed for t_bin index {index} at r={phi_discrete_r[np.argmax(diff)]}"

