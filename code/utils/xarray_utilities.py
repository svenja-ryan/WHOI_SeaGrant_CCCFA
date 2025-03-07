#######################################################
# Miscellaneous useful tools for xarray
#######################################################


'''
 Miscellaneous useful tools for xarray
'''
'''

    1) mean_weighted                 Weighted mean, e.g. for spatial mean with varying grid cell sizes
    2) rolling_xcorr                 Rolling cross-correlation
    3) xr_lagged_pearson             Lagged (time) pearson correlation
    4) deseason                      Remove seasonal cycle from timeseries

'''
'''
Updated November 4, 2022
'''



import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from numpy.polynomial.polynomial import polyval,polyfit
import matplotlib.dates as dates
import xarray as xr
import scipy.stats as stats




#
#
#--------------------------------------------------------------------------
# 1) derive weighted mean if weights are provides, normal mean otherwise
def mean_weighted(self, dim=None, weights=None):
    if weights is None:
        return self.mean(dim)
    else:
        return (self * weights).sum(dim) / weights.sum(dim)

#
#
#--------------------------------------------------------------------------
# 2) rolling cross-correlation    
def rolling_xcorr(ds,var,window):
    """
    Function to perform moving window cross-correlation
    
    As I haven't found another way yet, the xarray data is converted into pandas array 
    and back to xarray at the end.
    Interpolates across NaNs
    
    INPUT:
    ds = xarray dataset with both variables
    var = array with both variable names
    window = window size as integer
    
    OUTPUT:
    rolling_r = vector with rolling correlation coefficient (padded with nan at sides)
    
    """
    
    # separate both time series, convert to pandas and interpolate in case of nans
    var1 = ds[var[0]].to_dataframe().interpolate()
    var2 = ds[var[1]].to_dataframe().interpolate()

    # rolling window correlation
    # Compute rolling window synchrony
    rolling_r = var1[var[0]].rolling(window=window, center=True).corr(var2[var[1]])
    
    return rolling_r




#
#
#--------------------------------------------------------------------------
# 3) lagged pearson correlation
def xr_lagged_pearson(x,y,lags):
    """
    Derive lagged (in time) pearson correlation and significance
    
    INPUT
    x,y: Two timeseries (numpy) for cross-correlaction, for x=y autocorrelation
    lag: time lags in units of timestep, e.g. 1=1 month lag for monthly data
    
    OUTPUT
    cor: correlation coefficients for each lag
    p:   p-value for correlation at each lag
    """
    
    cor=[]
    p=[]

    for lag in lags:
        x1 = x[~np.isnan(x.values)]
        y1 = y.shift(time=lag).values[~np.isnan(x.values)]  # if want to use numpy array shift won't work
        x1 = x1[~np.isnan(y1)]
        y1 = y1[~np.isnan(y1)]

        if len(x1)>10:
            dummy1,dummy2 = stats.pearsonr(x1,y1) #crosscor(discharge_point,sss_point.shift(time=lag))#
            cor.append(dummy1)
            p.append(dummy2)
        else:
            cor.append(np.nan)
            p.append(np.nan)
    
    return cor,p
##############################################################


#
#
#--------------------------------------------------------------------------
# 4) remove seasonal cycle from time series
def deseason(ds,dt='month',timevar='time_counter',refperiod=None):
    dummy = timevar + '.' + dt
    if refperiod:
        if timevar=='time_counter':
            ds = ds.groupby(dummy)-ds.sel(time_counter=slice(*refperiod)).groupby(dummy).mean(timevar)        
        elif timevar=='time':
            ds = ds.groupby(dummy)-ds.sel(time=slice(*refperiod)).groupby(dummy).mean(timevar)
    else:
        ds = ds.groupby(dummy)-ds.groupby(dummy).mean(timevar)
    return ds

#
#
#----------------------------------------------------------------------------------
# 6) cross correlation using stats.personnr
def crosscor(x,y):
    x1 = x.values[~np.isnan(x.values)]
    y1 = y.values[~np.isnan(x.values)]
    x1 = x1[~np.isnan(y1)]
    y1 = y1[~np.isnan(y1)]

    return stats.pearsonr(x1,y1)