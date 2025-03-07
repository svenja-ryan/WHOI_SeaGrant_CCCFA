'''
 Different functions for Marine Heatwave/Cold spell detection in ORCA
'''
'''
    1) detect_events           # creates boolean array for MHWs and MCSs
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
#--------------------------------------------
# Helper functions
def twotailperc(data,alpha):
            percvalu = np.nanpercentile(data,100-alpha,axis=0)
            percvall = np.nanpercentile(data,alpha,axis=0)
            return percvalu,percvall
#
#
#
#
# 1) detect_events
#--------------------------------------------
# Create boolean array for MHW/cold spells

def detect_events(k003,pval,depth=0,opt='monthly'):

    if opt=='const':
        # detect events: constant threshold
        # significance upper, lower level
        percvalu,percvall = twotailperc(k003['votemper'][:,depth],pval)
        print('calculate constant threshold')

    elif opt=='seas':
        # seasonal,monthly threshold
        percvalu = np.empty([4,])
        percvall = np.empty([4,])
        dummy = k003.time_counter.dt.season
        for season,i in zip(['DJF','MAM','JJA','SON'],range(4)):
            test = k003.votemper[k003.time_counter.dt.season==season,0]
            percvalu[i] = np.nanpercentile(test,100-pval,axis=0)
            percvall[i] = np.nanpercentile(test,pval,axis=0)
            dummy[dummy==season]=percvalu[i]
            dummy=dummy.astype(np.float)

    elif opt=='monthly' :       
        percvalu = np.empty([12,])
        percvall = np.empty([12,])
        for month in np.arange(0,12):
            test = k003.votemper[k003.time_counter.dt.month==month+1,0]
            percvalu[month] = np.nanpercentile(test,100-pval,axis=0)
            percvall[month] = np.nanpercentile(test,pval,axis=0)

        # multiply seasonal cycle to match timeseries
        timelen = len(np.unique(k003.time_counter.dt.year))
        percvalu = np.tile(percvalu,timelen)
        percvall = np.tile(percvall,timelen)
        print('calculate monthly threshold')

    # create boolean-type vector to separate positive and negative events
    bool_mhw = (k003['votemper'][:,0]-percvalu)
    bool_mhw[bool_mhw>=0] = 1
    bool_mhw[bool_mhw<0] = np.NaN

    bool_mcw = (k003['votemper'][:,0]-percvall)
    bool_mcw[bool_mcw>0] = np.NaN
    bool_mcw[bool_mcw<=0] = 1
    
    return bool_mhw,bool_mcw