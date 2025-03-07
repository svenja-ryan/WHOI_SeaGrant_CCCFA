# # Perform EOF analysis on subset of ORCA data
# 
# I am using the eofs package by https://ajdawson.github.io/eofs/latest/index.html


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmocean as cmo
import datetime

from netCDF4 import Dataset, num2date
from eofs.xarray import Eof
from orca_utilities import cut_latlon_box, deseason_month, map_stuff, anomaly
from scipy import signal

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# perform eof and plot if wanted
def eof_orca_latlon_box(run,var,modes,lon_bnds,lat_bnds,pathfile,plot,time,eoftype):
    
    if (var=='temp'):
        key = 'votemper'
        key1 = "votemper"
    elif (var=='sal'):
        key = 'vosaline'
        key1 = "vosaline"
    elif (var=='MLD'):
        key = 'somxl010'
        key1 = "somxl010"
    
    # read data
    ds = xr.open_dataset(pathfile)
    #ds["time_counter"] = ds['time_counter']+(np.datetime64('0002-01-01')-np.datetime64('0001-01-01'))

    if time=='comparison':
        ds = ds.sel(time_counter=slice('1958-01-01', '2006-12-31'))
    
    
    # cut box for EOF at surface
    if var=='MLD':
        data = ds[key].sel(lon=slice(lon_bnds[0],lon_bnds[1]),
                       lat=slice(lat_bnds[0],lat_bnds[1]))
        #data = cut_latlon_box(ds[key][:,:,:],ds.lon,ds.lat,
                         # lon_bnds,lat_bnds)
    else:
        data = ds[key][:,0,:,:].sel(lon=slice(lon_bnds[0],lon_bnds[1]),
                       lat=slice(lat_bnds[0],lat_bnds[1]))
        #data = cut_latlon_box(ds[key][:,0,:,:],ds.lon,ds.lat,
                         # lon_bnds,lat_bnds)
    data=data.to_dataset()                 
    # detrend data
    data[key1]=(['time_counter', 'lat', 'lon'],signal.detrend(data[key].fillna(0),axis=0,
                                                                 type='linear'))
    
    #data=data.where(data!=0)
    
    # remove seasonal cycle and drop unnecessary coordinates
    if 'time_centered' in list(data.coords):
        data = deseason_month(data).drop('month').drop('time_centered') # somehow pca doesn't work otherwise
    else:
        data = deseason_month(data).drop('month') # somehow pca doesn't work otherwise
    
    # set 0 values back to nan
    data = data.where(data!=0)
    
    # EOF analysis
    #Square-root of cosine of latitude weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(data['lat'].values))
    coslat,_ = np.meshgrid(coslat,np.arange(0,len(data['lon'])))
    wgts = np.sqrt(coslat)
    solver = Eof(data[key],weights=wgts.transpose())
    pcs = solver.pcs(npcs=modes, pcscaling=1)
    if eoftype=='correlation':
        eof = solver.eofsAsCorrelation(neofs=modes)
    elif eoftype=='covariance': 
        eof = solver.eofsAsCovariance(neofs=modes)
    else:
        eof = solver.eofs(neofs=modes)
    varfr = solver.varianceFraction(neigs=4)
    print(varfr)

    #----------- Plotting --------------------
    plt.close("all")
    if plot ==1:
        for i in np.arange(0,modes):
            fig = plt.figure(figsize=(8,2))
            ax1 = fig.add_axes([0.1, 0.1, 0.3, 0.9],projection=ccrs.PlateCarree()) # main axes
            ax1.set_extent((lon_bnds[0],lon_bnds[1],lat_bnds[0],lat_bnds[1]))
            # discrete colormap
            cmap = plt.get_cmap('RdYlBu',len(np.arange(10,30))-1)    #inferno similar to cmo thermal
            eof[i,:,:].plot(ax=ax1,cbar_kwargs={'label': 'Correlation'},transform=ccrs.PlateCarree(),
                                         x='lon', y='lat', add_colorbar=True,
                                         cmap=cmap)
            gl = map_stuff(ax1)
            gl.xlocator = mticker.FixedLocator([100,110,120])
            gl.ylocator = mticker.FixedLocator(np.arange(-35,-10,5))
            plt.text(116,-24,str(np.round(varfr[i].values,decimals=2)), horizontalalignment='center',
            verticalalignment='center',transform=ccrs.PlateCarree(),fontsize=8)

            ax2 = fig.add_axes([0.5, 0.1, 0.55, 0.9]) # main axes
            plt.plot(pcs.time_counter,pcs[:,i].values,linewidth=0.1,color='k')
            anomaly(ax2,pcs.time_counter.values,pcs.values[:,i],[0,0])
            ax2.set_xlim([pcs.time_counter[0].values, pcs.time_counter[-1].values])
            plt.savefig(pathplots + 'eof_as' + eoftype +'_mode' + str(i)+ '_' + time + '_' + run + '_' + var + '.png',
                        dpi=300,bbox_inches = 'tight', pad_inches = 0.1)
            plt.show()
    #----------------------------------------------

    return pcs, eof, varfr
