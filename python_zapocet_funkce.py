#!/usr/bin/env python
# coding: utf-8

# In[1]:


# only few python packages are required:
import numpy as np  # working with arrays
import xarray as xr  # working with xarrays
import xrscipy.signal as dsp  # scipy for xarrays
#import xrscipy.signal.extra as dsp_extra
import matplotlib.pyplot as plt  # creating plots
import pandas as pd  # working with padnas frames, loading data from golem homepage
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
#from scipy.integrate import simps
import requests
from io import StringIO
from pathlib import Path

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

get_ipython().run_line_magic('matplotlib', 'inline')

# following code is used to predefine fonts styles and sizes in all the plots
from matplotlib import rc,rcParams

rc('font', weight='bold')
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams.update({'font.size': 12, 'axes.labelsize':14})


# In[2]:


#DATA ACQUISTION

def get_dataset(shotlist, rs, alpha, Te_shift, BPP_alpha, R_BPP, R_LP):
    """ 
    Function for loading the probe data and plasma current (I_p) from GOLEM shot homepage.
    Parameters
    ----------
    shotlist(List): python list with discharge numbers
    rs(List): python list with probe radial position
    Returns
    ----------
    ds(xr.Dataset): xarray Dataset with all the data and corresponding coordinates (t, r)
    """
    
    ds_container = []
    default_t = None
    no_bt_shots = {47020, 47021, 47022}  # Shots with unavailable Bt data
    data_dir = Path('probe data download')
    data_dir.mkdir(parents=True, exist_ok=True)

    for shot in shotlist:
        try:
            
            '''osc_data = pd.read_csv(
                f'http://golem.fjfi.cvut.cz/shots/{shot}/Devices/Oscilloscopes/TektrMSO64-a/TektrMSO64_ALL.csv',
                skiprows=10
            )'''
            
            # Load floating potential and bias probe data.
            # Reuse local CSV if already downloaded; otherwise download and cache it.
            osc_csv_path = data_dir / f'{shot}.csv'
            if osc_csv_path.exists():
                osc_data = pd.read_csv(osc_csv_path)
            else:
                osc_url = (
                    f'http://golem.fjfi.cvut.cz/shots/{shot}/Devices/Oscilloscopes/'
                    f'TektrMSO64-a/TektrMSO64_ALL.csv'
                )
                response = requests.get(osc_url, timeout=30)
                response.raise_for_status()
                osc_data = pd.read_csv(StringIO(response.text), skiprows=10)
                osc_data.to_csv(osc_csv_path, index=False)
            
            # Load Bt data only if the shot is not in the excluded list
            if shot in no_bt_shots:
                bt_data = None
            else:
                bt_data = pd.read_csv(
                    f'http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/BasicDiagnostics/Results/Bt.csv', 
                    names=['t_mag', 'mag_data']
                )
                        
        except Exception as e:
            print(f"Shot {shot} not loaded: {e}")
            continue

        # Initialize default time coordinate based on the first shot
        if default_t is None:
            default_t = osc_data['TIME'] * 1e3  # Convert time to ms
        
        # Interpolate probe data to the common time coordinate
        '''LP_1 = xr.DataArray(osc_data['CH3'] * R_LP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)
        LP_2 = xr.DataArray(osc_data['CH4'] * R_LP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)
        LP = (LP_1 + LP_2) / 2  # Average of LP_1 and LP_2

        BPP_1 = xr.DataArray(osc_data['CH2'] * R_BPP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)
        BPP_2 = xr.DataArray(osc_data['CH1'] * R_BPP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)
        BPP = (BPP_1 + BPP_2) / 2  # Average of BPP_1 and BPP_2'''


        BPP = xr.DataArray(osc_data['CH1'] * R_BPP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)
        LP = xr.DataArray(osc_data['CH4'] * R_LP, dims=['t'], coords={'t': osc_data['TIME'] * 1e3}).interp(t=default_t)

        if bt_data is None:
            Bt = None
        else:
            Bt = xr.DataArray(bt_data['mag_data'], dims=['t'], coords={'t': bt_data['t_mag']}).interp(t=default_t)
        
        # Remove DC offsets
        LP -= LP.sel(t=slice(0,1)).mean('t')
        BPP -= BPP.sel(t=slice(0,1)).mean('t')
        if Bt is not None:
            Bt -= Bt.sel(t=slice(0,1)).mean('t')
        
        # Create dataset for this shot
        ds_dict = {'Ubpp': BPP, 'Ulp': LP}
        if Bt is not None:
            ds_dict['Bt'] = Bt
        
        ds = xr.Dataset(ds_dict)
        ds_container.append(ds)

    # Combine data from all shots along the radial coordinate
    ds = xr.concat(ds_container, pd.Index(rs, name='r'))
    ds['Te'] = (ds['Ubpp'] - ds['Ulp']) / alpha
    ds['phi'] = ds['Ubpp'] + BPP_alpha * ds['Te']
    ds['Bt'] = Bt
    
    # LP shift
    # because probes are on a bit different radial coordinate, we sometimes need to manually adjust
    # the shift. For this, we have to interpolate one of the probes
    r_LP = ds.r - Te_shift
    LP_da = xr.DataArray(ds['Ulp'].data,dims=['r','t'],coords={'r':r_LP,'t':ds.t})
    LP_intp = LP_da.interp(r=ds.r,method='linear')
    Te_shifted = xr.DataArray((ds['Ubpp']-LP_intp)/alpha,dims=['r','t'], coords={'r':ds.r,'t':ds.t})
    ds['Te_shifted'] = Te_shifted
    ds['Te_shifted'] =ds['Te_shifted'].fillna(0)
    ds['Ulp_intp'] = LP_intp
    
    return ds



#--------------------------------------------------------------------------------------------------------------------------------

def get_plasma_times(shotlist):
    skipped_shots = {46013, 46015, 47020, 47021, 47022, 47024, #47020-47022 nedostupny basic, 42 2x neprorazilo = posledni 4 vyboje nepouzitelny pro extrakci t a Ip, ale data ze sond jsou
                     47037, 47038, 47039, 47040, 47041, 47042, 47044, 47045, 47046, 47047, #300 no plasma
                     47026, 47027, 47028, 47029, 47030, 47031, 47032, 47033, 47034, 47035,
                     45998}  #350 no plasma
    t_start_values = []
    t_end_values = []
    
    ip_mean_per_shot_values=[]
    
    for shot in shotlist:
        if shot in skipped_shots:
            continue  # Skip these shots entirely
        
        try:
            t_start_data = pd.read_csv(
                f'http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/PlasmaDetection/Results/t_plasma_start',
                header=None,
                names=["Number"]
            )
            t_start = t_start_data.iloc[0, 0]
            t_start_values.append(t_start)
            
            t_end_data = pd.read_csv(
                f'http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/PlasmaDetection/Results/t_plasma_end',
                header=None,
                names=["Number"]
            )
            t_end = t_end_data.iloc[0, 0]
            t_end_values.append(t_end)
        
        except Exception as e:
            print(f"Error processing shot {shot}: {e}")
    
    t1 = 0.5 * np.ceil(np.mean(t_start_values) / 0.5)
    t2 = 0.5 * np.floor(np.mean(t_end_values) / 0.5)
    
    return t1, t2


def add_bin_centers(ds, bin_dim='t_bins', new_dim_name='t'):
    """
    Adds a new coordinate with bin centers to an xarray Dataset.
    
    Parameters:
    - ds (xarray.Dataset): The Dataset with grouped bins.
    - bin_dim (str): The name of the dimension that contains the bins (e.g., 'rec_bins').
    - new_dim_name (str): The name of the new coordinate that will hold the bin centers.

    Returns:
    - xarray.Dataset: A new Dataset with the bin centers as a coordinate.
    """
    # Extract the bin intervals from the specified dimension
    bin_intervals = ds[bin_dim].values
    
    # Calculate the centers of each bin
    bin_centers = [(interval.left + interval.right) / 2 for interval in bin_intervals]
    
    # Assign the bin centers as a new coordinate in the dataset
    ds = ds.assign_coords({new_dim_name: bin_centers})
    
    return ds

def process_dataset(shotlist, rs, alpha, Te_shift, BPP_alpha, R_BPP, R_LP, t1, t2, step):
    """Load dataset, create time bins, and compute statistics."""
    t_bins = np.arange(t1, t2, step)
    
    ds = get_dataset(shotlist, rs, alpha, Te_shift, BPP_alpha, R_BPP, R_LP)
    gb_ds = ds.groupby_bins('t', t_bins)
    
    ds_mean = gb_ds.mean('t')
    ds_std = gb_ds.std('t')

    ds_mean = add_bin_centers(ds_mean)
    
    return ds_mean, ds_std, t_bins


def fit_profile_gauss_regression(ds, ds_std, C_min, C_max, RBF_min, RBF_max, noise_min, noise_max, n_samples):
    '''
    Function for fitting the Gaussian Process Regression to the data.
    Parameters
    ----------
    ds(xr.DataArray): xarray DataArray with the data to fit.
    ds_std(xr.DataArray): xarray DataArray with the standard deviation of the data.
    C_min(float): minimum value for the constant kernel.
    C_max(float): maximum value for the constant kernel.
    RBF_min(float): minimum value for the RBF kernel.
    RBF_max(float): maximum value for the RBF kernel.
    noise_min(float): minimum value for the noise kernel.
    noise_max(float): maximum value for the noise kernel.
    Returns
    ----------
    da(xr.DataArray): xarray DataArray with the fitted data.
    da_std(xr.DataArray): xarray DataArray with the standard deviation of the fitted data.
    '''
    # Extract the data you want to fit from gb_phi
    r = ds.r.values  # the independent variable
    y = ds.values  # the dependent variable
    y_err = ds_std.values  # error bars (standard deviation)
    
    # Create more points for the final plot
    x_high_res = np.linspace(r.min().item(), r.max().item(), 100)

    # Gaussian Process kernel: Constant * RBF + WhiteKernel
    # RBF: models the smooth variation
    # WhiteKernel: models observation noise, with noise level based on `y_err`
    kernel = (C(1.0, (C_min, C_max)) 
              * RBF(length_scale=1e1, length_scale_bounds=(RBF_min, RBF_max))
#               + WhiteKernel(noise_level=0.01, noise_level_bounds=(noise_min, noise_max))
             )
    # Instantiate and fit the GaussianProcessRegressor (no need for alpha now)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha = y_err)

    gp.fit(r[:, np.newaxis], y)

    # Predict using the fitted GPR model for high-resolution x values
    y_pred, sigma = gp.predict(x_high_res[:, np.newaxis], return_std=True, )
    
    da = xr.DataArray(y_pred, dims=['r'], coords={'r': x_high_res})
    da_std = xr.DataArray(sigma, dims=['r'], coords={'r': x_high_res})
    
    
    samples = gp.sample_y(x_high_res[:, np.newaxis], n_samples = n_samples)
    
    # Convert to xarray for convenience
    realizations = xr.DataArray(samples, dims=['r', 'realization'], coords={'r': x_high_res, 'realization': range(n_samples)})

    return da, da_std, realizations, gp.kernel_



def compute_phi_Er_vpol_omega_s_fit(ds_mean, ds_std, to_cycle, 
                                C_min_phi, C_max_phi, RBF_min_phi, RBF_max_phi,
                                noise_min, noise_max, n_samples):
    """Perform Gaussian regression, compute Er, poloidal velocity (Vθ), and shearing rate (ωs)."""
    
    Ers_mean, Ers_ci_lower, Ers_ci_upper = [], [], []
    v_pol_mean, v_pol_ci_lower, v_pol_ci_upper = [], [], []
    omega_s_mean, omega_s_ci_lower, omega_s_ci_upper = [], [], []
    all_phi_values, all_phi_errors = [], []

            
    for index in range(to_cycle):
        # Gaussian regression
        da_phi, da_phi_std, realizations_phi, kernel_info = fit_profile_gauss_regression(
            ds_mean['phi'].isel(t_bins=index),
            ds_std['phi'].isel(t_bins=index),
            C_min_phi, C_max_phi, RBF_min_phi, RBF_max_phi, noise_min, noise_max, n_samples
        )
        
        phi_data = ds_mean['phi'].isel(t_bins=index)
        phi_std = ds_std['phi'].isel(t_bins=index)
        all_phi_values.extend(phi_data.values)
        all_phi_errors.extend(phi_std.values)
        
        bt_data = ds_mean['Bt'].isel(t_bins=index)
        
        # Er
        Er_tmp = -realizations_phi.differentiate('r')
        Er_tmp_mean = Er_tmp.mean('realization')
        Er_tmp_std = (Er_tmp - Er_tmp_mean).std('realization')

        ci_lower1 = Er_tmp_mean - 1.96 * Er_tmp_std
        ci_upper1 = Er_tmp_mean + 1.96 * Er_tmp_std

        Ers_mean.append(Er_tmp_mean.expand_dims(t=[ds_mean.t[index].item()]))
        Ers_ci_lower.append(ci_lower1.expand_dims(t=[ds_mean.t[index].item()]))
        Ers_ci_upper.append(ci_upper1.expand_dims(t=[ds_mean.t[index].item()]))

        # Vθ
        v_pol_tmp = Er_tmp / bt_data
        v_pol_tmp_mean = v_pol_tmp.mean('realization')
        v_pol_tmp_std = (v_pol_tmp - v_pol_tmp_mean).std('realization')

        ci_lower2 = v_pol_tmp_mean - 1.96 * v_pol_tmp_std
        ci_upper2 = v_pol_tmp_mean + 1.96 * v_pol_tmp_std

        v_pol_mean.append(v_pol_tmp_mean)
        v_pol_ci_lower.append(ci_lower2)
        v_pol_ci_upper.append(ci_upper2)

        # ωs
        omega_s_tmp = v_pol_tmp.differentiate('r')
        omega_s_tmp_mean = omega_s_tmp.mean('realization')
        omega_s_tmp_std = (omega_s_tmp - omega_s_tmp_mean).std('realization')

        ci_lower3 = omega_s_tmp_mean - 1.96 * omega_s_tmp_std
        ci_upper3 = omega_s_tmp_mean + 1.96 * omega_s_tmp_std

        omega_s_mean.append(omega_s_tmp_mean)
        omega_s_ci_lower.append(ci_lower3)
        omega_s_ci_upper.append(ci_upper3)
        
    
    return (
        Ers_mean, Ers_ci_lower, Ers_ci_upper, 
        v_pol_mean, v_pol_ci_lower, v_pol_ci_upper, 
        omega_s_mean, omega_s_ci_lower, omega_s_ci_upper,
        all_phi_values, all_phi_errors
        )


def compute_phi_Er_vpol_omega_s_discrete(ds_mean, ds_std, to_cycle):
    """Compute Er, poloidal velocity (Vtheta), and shearing rate (omega s) from discrete phi points."""

    Ers_mean, Ers_ci_lower, Ers_ci_upper = [], [], []
    v_pol_mean, v_pol_ci_lower, v_pol_ci_upper = [], [], []
    omega_s_mean, omega_s_ci_lower, omega_s_ci_upper = [], [], []
    all_phi_values, all_phi_errors = [], []

    for index in range(to_cycle):
        phi_data = ds_mean['phi'].isel(t_bins=index)
        phi_std = ds_std['phi'].isel(t_bins=index)
        all_phi_values.extend(phi_data.values)
        all_phi_errors.extend(phi_std.values)

        bt_data = ds_mean['Bt'].isel(t_bins=index)

        # Er
        Er_tmp = -phi_data.differentiate('r')
        Er_tmp_std = phi_std.differentiate('r')

        ci_lower1 = Er_tmp - 1.96 * Er_tmp_std
        ci_upper1 = Er_tmp + 1.96 * Er_tmp_std

        Ers_mean.append(Er_tmp.expand_dims(t=[ds_mean.t[index].item()]))
        Ers_ci_lower.append(ci_lower1.expand_dims(t=[ds_mean.t[index].item()]))
        Ers_ci_upper.append(ci_upper1.expand_dims(t=[ds_mean.t[index].item()]))

        # Vtheta
        v_pol_tmp = Er_tmp / bt_data
        v_pol_tmp_std = Er_tmp_std / bt_data

        ci_lower2 = v_pol_tmp - 1.96 * v_pol_tmp_std
        ci_upper2 = v_pol_tmp + 1.96 * v_pol_tmp_std

        v_pol_mean.append(v_pol_tmp)
        v_pol_ci_lower.append(ci_lower2)
        v_pol_ci_upper.append(ci_upper2)

        # omega s
        omega_s_tmp = v_pol_tmp.differentiate('r')
        omega_s_tmp_std = v_pol_tmp_std.differentiate('r')

        ci_lower3 = omega_s_tmp - 1.96 * omega_s_tmp_std
        ci_upper3 = omega_s_tmp + 1.96 * omega_s_tmp_std

        omega_s_mean.append(omega_s_tmp)
        omega_s_ci_lower.append(ci_lower3)
        omega_s_ci_upper.append(ci_upper3)

    return (
        Ers_mean, Ers_ci_lower, Ers_ci_upper,
        v_pol_mean, v_pol_ci_lower, v_pol_ci_upper,
        omega_s_mean, omega_s_ci_lower, omega_s_ci_upper,
        all_phi_values, all_phi_errors
    )


def plot_phi_discrete_and_fit(ax, ds_mean, ds_std, t_bins, step, t1, to_cycle, C_min_phi, C_max_phi, RBF_min_phi, RBF_max_phi,
                                   noise_min, noise_max, n_samples, color_map):
    """Plot Phi values with Gaussian regression and confidence intervals."""
    colors = [color_map(i) for i in np.linspace(0, 1, to_cycle)]  # Generate distinct colors

    for index in range(to_cycle):
        t_start = t1 + index * step
        t_end = t_start + step
        time_label = f"{t_start}-{t_end} ms"

        current_color = colors[index]  # Assign unique color for each time bin

        da, da_std, realizations, kernel_info = fit_profile_gauss_regression(
            ds_mean['phi'].isel(t_bins=index),
            ds_std['phi'].isel(t_bins=index),
            C_min_phi, C_max_phi,
            RBF_min_phi, RBF_max_phi,
            noise_min, noise_max, n_samples
        )

        ax.errorbar(
            ds_mean['phi'].isel(t_bins=index).r.data,
            ds_mean['phi'].isel(t_bins=index).data,
            ds_std['phi'].isel(t_bins=index).data,
            elinewidth=0.5, capsize=2.5,
            color=current_color, marker='s', ls='' 
        )

        da.plot(ax=ax, color=current_color, label=f'{time_label}')

        ax.fill_between(da.r, da.values - 1.96 * da_std.values, da.values + 1.96 * da_std.values,
                     alpha=0.1, color=current_color)


    ax.grid(True)
    ax.set_xlabel('$r$ [mm]')
    ax.set_ylabel(r'$\phi$ [V]')
    ax.set_title(f'plasma potential profile, discrete + fit', fontsize=14, fontweight='bold')
    ax.legend()


def plot_er_fit(ax, Er_mean, Er_ci_lower, Er_ci_upper, t_bins, step, t1, to_cycle):
    """Plot radial electric field Er with confidence intervals."""
    ax.set_prop_cycle(color=[LinearSegmentedColormap.from_list("custom_blue_red", ['blue', 'red'])(i) for i in np.linspace(0, 1, to_cycle)])

    for index in range(to_cycle):
        t_start = t1 + index * step
        t_end = t_start + step
        time_label = f"{t_start}-{t_end} ms"

        Er_mean_time = Er_mean.isel(t=index)
        ci_lower_time = Er_ci_lower.isel(t=index)
        ci_upper_time = Er_ci_upper.isel(t=index)

        ax.plot(Er_mean_time.r, Er_mean_time)
        ax.fill_between(Er_mean_time.r, ci_lower_time, ci_upper_time, alpha=0.1)

    ax.grid(True)
    ax.set_xlabel(r'$r$ [mm]')
    ax.set_ylabel(r'$E_\mathrm{r}$ [kV/m]')
    ax.set_title(f'radial electric field profile, fit', fontsize=14, fontweight='bold')


def plot_er_discrete(ax, Er_mean, Er_ci_lower, Er_ci_upper, t_bins, step, t1, to_cycle):
    """Plot radial electric field Er from discrete points with confidence intervals."""
    ax.set_prop_cycle(color=[LinearSegmentedColormap.from_list("custom_blue_red", ['blue', 'red'])(i) for i in np.linspace(0, 1, to_cycle)])

    for index in range(to_cycle):
        t_start = t1 + index * step
        t_end = t_start + step
        time_label = f"{t_start}-{t_end} ms"

        Er_mean_time = Er_mean.isel(t=index)
        ci_lower_time = Er_ci_lower.isel(t=index)
        ci_upper_time = Er_ci_upper.isel(t=index)

        yerr = np.vstack([
            np.abs((Er_mean_time - ci_lower_time).values),
            np.abs((ci_upper_time - Er_mean_time).values)
        ])
        ax.errorbar(
            Er_mean_time.r.values,
            Er_mean_time.values,
            yerr=yerr,
            linestyle='-',
            capsize=2.5,
            elinewidth=0.5, marker='s'
        )

    ax.grid(True)
    ax.set_xlabel(r'$r$ [mm]')
    ax.set_ylabel(r'$E_\mathrm{r}$ [kV/m]')
    ax.set_title(f'radial electric field profile, discrete', fontsize=14, fontweight='bold')


def plot_omega_shearing_fit(ax, omega_s_mean, omega_s_ci_lower, omega_s_ci_upper, t_bins, step, t1, to_cycle):
    """Plot radial electric field Er with confidence intervals."""
    ax.set_prop_cycle(color=[LinearSegmentedColormap.from_list("custom_blue_red", ['blue', 'red'])(i) for i in np.linspace(0, 1, to_cycle)])

    for index in range(to_cycle):
        t_start = t1 + index * step
        t_end = t_start + step
        time_label = f"{t_start}-{t_end} ms"

        omega_s_mean_time = omega_s_mean.isel(t=index)
        omega_s_ci_lower_time = omega_s_ci_lower.isel(t=index)
        omega_s_ci_upper_time = omega_s_ci_upper.isel(t=index)

        ax.plot(omega_s_mean_time.r, omega_s_mean_time)
        ax.fill_between(omega_s_mean_time.r, omega_s_ci_lower_time, omega_s_ci_upper_time, alpha=0.1)

    ax.grid(True)
    ax.set_xlabel(r'$r$ [mm]')
    ax.set_ylabel(r'$\omega_\mathrm{E \times B}$ [10$^6$ s$^{-1}$]')
    ax.set_title(f'shearing rate radial profile, fit', fontsize=14, fontweight='bold')

def plot_omega_shearing_discrete(ax, omega_s_mean, omega_s_ci_lower, omega_s_ci_upper, t_bins, step, t1, to_cycle):
    """Plot shearing rate from discrete points with confidence intervals."""
    ax.set_prop_cycle(color=[LinearSegmentedColormap.from_list("custom_blue_red", ['blue', 'red'])(i) for i in np.linspace(0, 1, to_cycle)])

    for index in range(to_cycle):
        t_start = t1 + index * step
        t_end = t_start + step
        time_label = f"{t_start}-{t_end} ms"

        omega_s_mean_time = omega_s_mean.isel(t=index)
        ci_lower_time = omega_s_ci_lower.isel(t=index)
        ci_upper_time = omega_s_ci_upper.isel(t=index)

        yerr = np.vstack([
            np.abs((omega_s_mean_time - ci_lower_time).values),
            np.abs((ci_upper_time - omega_s_mean_time).values)
        ])
        ax.errorbar(
            omega_s_mean_time.r.values,
            omega_s_mean_time.values,
            yerr=yerr,
            linestyle='-',
            capsize=2.5,
            elinewidth=0.5, marker='s'
        )

    ax.grid(True)
    ax.set_xlabel(r'$r$ [mm]')
    ax.set_ylabel(r'$\omega_\mathrm{E \times B}$ [10$^6$ s$^{-1}$]')
    ax.set_title(f'shearing rate radial profile, discrete', fontsize=14, fontweight='bold')
