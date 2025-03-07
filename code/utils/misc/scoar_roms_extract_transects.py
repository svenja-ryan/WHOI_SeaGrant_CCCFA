#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Extract transects from SCOAR/ROMS model output.


'''

# Imports --------------------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr
import xroms

from datetime import datetime
from dask_jobqueue import SLURMCluster
from distributed import Client
from functools import partial
from pathlib import Path

# Parameters -----------------------------------------------------------------

# SCOAR simulation
# scoarID = 'r01_wfp_ww32roms'
scoarID = 'r02_nowfp_ww32roms'

# year
year = 2019
season = 'jja'

# line identifier
lineID = 'TEL' # Tuckerton Endurance Line

# output frequency
out_freq = '1d'

# method to calculate surface reference density
# ref_method = '5m_layer'
ref_method = 'surface_level'

# Data directories -----------------------------------------------------------

# SCOAR/ROMS output directory
roms_dir = Path(f'/vortexfs1/share/seolab/wfp/jjas/{year}_new_runs/')

# Path to .csv file with ROMS transect indices
proc_dir = Path('/vortexfs1/home/christoph.renkl/seolab/wfp/transect_indices')

# name and path of ROMS grid file
grid_file =  Path(f'/vortexfs1/share/seolab/wfp/jjas/GRID/roms-gs_highRes_new-grid_nolake.nc')

# Functions ------------------------------------------------------------------

def main():
    '''Extract ROMS output along transects.'''
    
    print(f'[{datetime.now()}] >> Loading data...')

    # ROMS `avg` output to `xarray.Dataset`
    ds = roms2ds(
        roms_dir/scoarID/'ROMS',
        'avg',
        [year],
        freq=out_freq,
        grid_file=grid_file
    )
   
    print(f'[{datetime.now()}] >> Calculating potential density...')

    # Calculate potential density
    ds['sig0'] = xroms.density(ds['temp'], ds['salt'], z=0) - 1000
    
    # mixed layer depth

    # path and file name
    mld_dir = roms_dir/scoarID/'ROMS'/'derived'
    mld_file = mld_dir/f'avg_{scoarID}_{out_freq}_mld_{ref_method}_{season}{year}.nc'

    # ROMS mixed layer depth to `xarray.DataArray`
    mld = roms_mld2da(mld_file)
    mld['ocean_time'] = ds['ocean_time']
    ds['mld'] = mld

    # variables to extract along transect
    variables = [
        'salt',
        'temp',
        'u_eastward',
        'v_northward',
        'w',
        'zeta',
        'sig0',
        'mld'
    ]

    # extract data along transect
    ds_out = roms_transect(ds[variables], lineID, proc_dir)

    print(f'[{datetime.now()}] >> Writing NetCDF file...')

    # write NetCDF file
    out_dir = roms_dir/scoarID/'ROMS'/'derived'/'transects'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir/f'avg_{scoarID}_{out_freq}_{lineID}_{season}{year}.nc'
    ds_out.to_netcdf(out_file)
    
    print('')
    print(f'[{datetime.now()}] >> FINISHED')

def roms_transect(roms, transect, obs_dir=Path('../data/processed')):

    # get ROMS indices
    inds = pd.read_csv(proc_dir / f'{lineID}_transect_SCOAR_wfp_indices.csv')

    xi_inds = xr.DataArray(
        inds["xi_ind"].values,
        dims=["dist"],
        coords={
            "dist": ("dist", inds["distance"].values)
        }
    )
    eta_inds = xr.DataArray(
        inds["eta_ind"].values,
        dims=["dist"],
        coords={
            "dist": ("dist", inds["distance"].values)}
    )
    
    # select model data along transect
    sec = roms.isel(eta_rho=eta_inds, xi_rho=xi_inds)
    
    # rotate currents
    if {'u_eastward', 'v_northward'}.issubset(sec.data_vars):
        
        # the variable `angle` the indices file is the azimuth, i.e., the angle
        # from north of the orientation of the transect ind degrees.
        theta = xr.DataArray(
            np.deg2rad(90 - inds['angle'].values),
            dims=['dist'],
            coords={ 
                'dist': ('dist', inds['distance'].values)
            }
        )

        sec['u_cross_sec'] = (
            sec['u_eastward'] * np.cos(theta)
            + sec['v_northward'] * np.sin(theta)
        ).variable

        sec['v_along_sec'] = (
            - sec['u_eastward'] * np.sin(theta)
            + sec['v_northward'] * np.cos(theta)
        ).variable
        
    return sec

def roms_mld2da(mld_file):
    ds = xr.open_dataset(mld_file)
    return ds['mld']


def roms2ds(roms_dir, ftype, years, freq='1h', grid_file=None):

    grid = roms_grid2ds(grid_file)

    preprocess_partial = partial(
        _preprocess,
        grid=grid
    ) 
    
    # list of files
    file_list = [
        sorted((roms_dir/ftype/f'{y}').glob(f'{ftype}_*{freq}*.nc'))
        for y in years
    ]

    # upack list of lists for each month and year
    # https://blog.finxter.com/join-list-of-lists/
    roms_files = [f for l in file_list for f in l]
    
    roms = xr.open_mfdataset(
        roms_files,
        preprocess=preprocess_partial,
        compat='override',
        combine='by_coords',
        data_vars='minimal',
        coords='minimal',
        parallel=True
    )
    
    return roms


def _preprocess(ds, grid):
    
    # add missing dimensions - hard coding parameters for now
    Vtransform = 2
    Vstretching = 4
    theta_s = 7.
    theta_b = 2
    hc = ds['hc']

    if 's_rho' in ds.dims:
        N = ds.sizes['s_rho']
    elif 's_w' in ds.dims:
        N = ds.sizes['s_w'] - 1

    if 'xi_u' not in ds.dims:
        ds['xi_u'] = np.arange(ds.sizes['xi_rho']-1)
    if 'eta_v' not in ds.dims:
        ds['eta_v'] = np.arange(ds.sizes['eta_rho']-1)

    if 'pm' not in ds.data_vars:
        ds['pm'] = grid['pm']
    if 'pn' not in ds.data_vars:
        ds['pn'] = grid['pn']

    if 'mask_rho' not in ds.data_vars:
        ds['mask_rho'] = grid['mask_rho']
    
    if 'Vtransform' not in ds.data_vars:
        ds['Vtransform'] = Vtransform
    
    if 'Cs_w' not in ds.coords:
        s_w, Cs_w = stretching(Vstretching, theta_s, theta_b, hc, N, kgrid = 1)
        ds['Cs_w'] = xr.DataArray(
            Cs_w, coords={'s_w': s_w}
        )

    if 'zeta' in ds.data_vars:

        # `xroms` pre-processing
        ds, _ = xroms.roms_dataset(ds, include_Z0=True)

    return ds


def roms_grid2ds(grid_file):
    return xr.open_dataset(grid_file)


def stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, report=False):
    """Compute ROMS vertical coordinate stretching function.

    Adapted from original Matlab code.

    Inputs
    ------
    Vstretching: int
        Vertical stretching function:
          Vstretching = 1,  original (Song and Haidvogel, 1994)
          Vstretching = 2,  A. Shchepetkin (UCLA-ROMS, 2005)
          Vstretching = 3,  R. Geyer BBL refinement
          Vstretching = 4,  A. Shchepetkin (UCLA-ROMS, 2010)
          Vstretching = 5,  Quadractic (Souza et al., 2015)
    theta_s: float
        S-coordinate surface control parameter.
    theta_b: float
        S-coordinate bottom control parameter.
    hc:
    N:
    kgrid:
    report:

    Returns
    -------
    s: array-like
        S-coordinate independent variable at vertical RHO- or W-points.
    C: array-like
        Nondimensional, monotonic, vertical stretching function.

    Example usage
    -------------
    >>> s, C = stretching(...)
    """

    # Check parameter Vstretching
    assert (Vstretching >= 1), 'Illegal parameter Vstretching = {}.'.format(
        Vstretching)
    assert (Vstretching <= 5), 'Illegal parameter Vstretching = {}.'.format(
        Vstretching)

    Np = N + 1

    # --------------------------------------------------------------------------
    # Compute ROMS S-coordinates vertical stretching function
    # --------------------------------------------------------------------------

    # Original vertical stretching function(Song and Haidvogel, 1994).

    if (Vstretching == 1):

        ds = 1.0/N
        if (kgrid == 1):
            Nlev = Np
            lev = np.arange(N+1)
            s = (lev - N) * ds
        else:
            Nlev = N
            lev = np.arange(0, N+1) - 0.5
            s = (lev - N) * ds

        if (theta_s > 0):
            Ptheta = np.sinh(theta_s * s) / np.sinh(theta_s)
            Rtheta = np.tanh(theta_s * (s+0.5)) / \
                (2.0 * np.tanh(0.5*theta_s)) - 0.5
            C = (1.0 - theta_b) * Ptheta + theta_b * Rtheta
        else:
            C = s

    # A. Shchepetkin(UCLA-ROMS, 2005) vertical stretching function.

    elif (Vstretching == 2):

        alfa = 1.0
        beta = 1.0
        ds = 1.0/N
        if (kgrid == 1):
            Nlev = Np
            lev = np.arange(N+1)
            s = (lev - N) * ds
        else:
            Nlev = N
            lev = np.arange(1, N+1) - 0.5
            s = (lev - N) * ds

        if (theta_s > 0):
            Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
            if (theta_b > 0):
                Cbot = -1.0 + np.sinh(theta_b * (s+1.0)) / np.sinh(theta_b)
                weigth = (s + 1.0)**alfa * \
                    (1.0 + (alfa/beta) * (1.0 - (s + 1.0)**beta))
                C = weigth * Csur + (1.0 - weigth) * Cbot
            else:
                C = Csur
        else:
            C = s

    # R. Geyer BBL vertical stretching function.

    elif (Vstretching == 3):

        ds = 1.0/N
        if (kgrid == 1):
            Nlev = Np
            lev = np.arange(N+1)
            s = (lev - N) * ds
        else:
            Nlev = N
            lev = np.arange(1, N+1) - 0.5
            s = (lev - N) * ds

        if (theta_s > 0):
            exp_s = theta_s  # surface stretching exponent
            exp_b = theta_b  # bottom  stretching exponent
            alpha = 3  # scale factor for all hyperbolic functions
            Cbot = np.log(np.cosh(alpha*(s+1)**exp_b)) / \
                np.log(np.cosh(alpha)) - 1
            Csur = -np.log(np.cosh(alpha*abs(s)**exp_s)) / \
                np.log(np.cosh(alpha))
            weight = (1 - np.tanh(alpha*(s + .5))) / 2
            C = weight * Cbot + (1 - weight) * Csur
        else:
            C = s

    # A. Shchepetkin(UCLA-ROMS, 2010) double vertical stretching function
    # with bottom refinement

    elif (Vstretching == 4):

        ds = 1.0/N
        if (kgrid == 1):
            Nlev = Np
            lev = np.arange(N+1)
            s = (lev - N) * ds
        else:
            Nlev = N
            lev = np.arange(1, N+1) - 0.5
            s = (lev-N) * ds

        if (theta_s > 0):
            Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
        else:
            Csur = -s**2

        if (theta_b > 0):
            Cbot = (np.exp(theta_b * Csur) - 1.0) / (1.0 - np.exp(-theta_b))
            C = Cbot
        else:
            C = Csur

    # Quadratic formulation to enhance surface exchange.
    #
    # (J. Souza, B.S. Powell, A.C. Castillo-Trujillo, and P. Flament, 2014:
    # The Vorticity Balance of the Ocean Surface in Hawaii from a
    # Regional Reanalysis.'' J. Phys. Oceanogr., 45, 424-440)

    elif (Vstretching == 5):

        if (kgrid == 1):
            Nlev = Np
            lev = np.arange(N+1)
            s = -(lev * lev - 2.0*lev*N + lev + N*N - N) / (N*N - N) \
                - 0.01 * (lev*lev - lev*N) / (1.0 - N)
            s[0] = -1.0
        else:
            Nlev = N
            lev = np.arange(1, N+1) - 0.5
            s = -(lev*lev - 2.0*lev*N + lev + N*N - N) / (N*N - N) \
                - 0.01 * (lev*lev - lev*N) / (1.0 - N)

        if (theta_s > 0):
            Csur = (1.0 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1.0)
        else:
            Csur = -s**2

        if (theta_b > 0):
            Cbot = (np.exp(theta_b * Csur) - 1.0) / (1.0 - np.exp(-theta_b))
            C = Cbot
        else:
            C = Csur

    if (report):
        print(' ')
        if (Vstretching == 1):
            print('Vstretching = {} Song and Haidvogel (1994)'.format(Vstretching))
        if (Vstretching == 2):
            print('Vstretching = {} Shchepetkin (2005)'.format(Vstretching))
        if (Vstretching == 3):
            print('Vstretching = {} Geyer (2009), BBL'.format(Vstretching))
        if (Vstretching == 4):
            print('Vstretching = {} Shchepetkin (2010)'.format(Vstretching))
        if (Vstretching == 5):
            print('Vstretching = {} Souza et al. (2014)'.format(Vstretching))

        if (kgrid == 1):
            print('   kgrid    = {}   at vertical W-points'.format(kgrid))
        else:
            print('   kgrid    = {}   at vertical RHO-points'.format(kgrid))

        print('   theta_s  = {}'.format(theta_s))
        print('   theta_b  = {}'.format(theta_b))
        print('   hc       = {}'.format(hc))
        print(' ')
        print(' S-coordinate curves: k, s(k), C(k)')
        print(' ')

        for k in range(Nlev, 0, -1):
            print('    {:3g}   {:20.12e}   {:20.12e}'.format(
                k, s[k-1], C[k-1]))

        print(' ')

    return s, C


if __name__ == '__main__':

    # connecting to cluster
    cluster = SLURMCluster(
        cores=36,
        processes=1,
        memory='192GB',
        walltime='01:00:00',
        queue='compute'
    )
    
    print(cluster.job_script())
    
    cluster.scale(2)
    client = Client(cluster)
    
    # main program
    main()
