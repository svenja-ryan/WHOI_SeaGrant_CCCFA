#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 08:15:14 2022

Functions for LRAUV data processing

@author: sryan
"""
import numpy as np
import xarray as xr


#########################################################################
# function to merge up and downcast of given section (and some manipulation)
def merge_casts(ds,varname='salinity'):
    
    plot=1

    # extract data along section
    z = ds['sea_water_pressure'].values
    lon = ds['lon_cor'].values
    lat = ds['lat_cor'].values
    time = ds['time'].values
    if varname=='salinity':
        var = ds['sea_water_salinity'].values
    elif varname=='temperature':
        var = ds['sea_water_temperature']
    elif varname=='density':
        var = ds['sea_water_pot_density']

    # take difference and find zero crossing
    dz = np.diff(z)
    # find values where drifting at surface and discart
    wo = np.where(abs(dz)>0.1)
    dz = dz[wo]
    ind0 = list(np.where(np.diff(np.sign(dz)))[0]+1)
    ind0.insert(0,0) # add zero at the beginning

    # apply to variables
    lon = lon[wo]
    lat = lat[wo]
    var = var[wo]
    z = z[wo]
    time = time[wo]


    ##############################
    # Do profile interpolation and merging
    ##############################

    # predefine empty arrays
    zint = np.arange(5,50,0.1)  # depth level for interpolation
    varint = np.ones((1500,len(zint)))*np.nan
    latint = np.ones(1500,)*np.nan
    lonint = np.ones(1500,)*np.nan
    timeint = np.ones(1500,)*np.nan
    dummy = np.ones((1500,2))*np.nan
    varmerged = np.ones((800,len(zint)))*np.nan
    latmerged = np.ones(800,)*np.nan
    lonmerged = np.ones(800,)*np.nan
    timemerged = [None]*800 # need a list for time strings

    j=0
    # inteprolate downcast (!!!! REMINDER that indexing is non inclusive of last index, e.g. no -1 needed)
    for i in np.arange(0,len(ind0)-2,2):
        profdown = var[ind0[i]:ind0[i+1]+1]
        zdown = z[ind0[i]:ind0[i+1]+1]
        if len(zdown)<3:
            profdown_int = np.ones(len(zint))*np.nan
        else:
#             profdown_int = np.interp(zint,zdown,profdown,right=np.nan,left=np.nan)
            f=interpolate.interp1d(zdown,profdown,fill_value='extrapolate')
            profdown_int = f(zint)
        varint[i,:] = profdown_int
        lonint[i] = np.mean(lon[[ind0[i],ind0[i+1]]]) # average coordinates from top and bottom of profile
        latint[i] = np.mean(lat[[ind0[i],ind0[i+1]]])
        timeint[i] = time[ind0[i]]

        # interpolate upcast
        profup = np.flip(var[ind0[i+1]:ind0[i+2]])
        zup = np.flip(z[ind0[i+1]:ind0[i+2]])
        if len(zup)<3:
            profup_int = np.ones(len(zint))*np.nan
        else:
#             profup_int = np.interp(zint,zup,profup,right=np.nan,left=np.nan)
            f=interpolate.interp1d(zup,profup,fill_value='extrapolate')
            profup_int = f(zint)
        #     sint = np.vstack((sint,profup_int.T))
        varint[i+1,:] = profup_int
        lonint[i+1] = np.nanmean(lon[[ind0[i+1],ind0[i+2]]])
        latint[i+1] = np.nanmean(lat[[ind0[i+1],ind0[i+2]]])
        timeint[i+1] = time[ind0[i+2]-1]

        # average between profiles and discard shallow profiles 
        dummy = np.mean((profdown_int,profup_int),axis=0) # change to nanmean if you want to keep all values and not cut off at common depth
        if (len(zup)>5) & (len(zdown)>5):
            if (max(zint[np.isfinite(dummy)])>20):
                varmerged[j,:] = dummy
                # average lat,lon
                lonmerged[j] = np.nanmean(lonint[i:i+2])
                latmerged[j] = np.nanmean(latint[i:i+2])
                timemerged[j] = np.datetime64(pd.to_datetime(np.average(timeint[i:i+1])))
#                 print(timemerged[j])
                j=j+1
#                 print(j)


    # remove empty columns
    varmerged = varmerged[0:j]
    lonmerged = lonmerged[0:j]
    latmerged = latmerged[0:j]
    timemerged = timemerged[0:j]
    
    # create an xarray
    section={}
    section['lat'] = latmerged
    section['lon'] = lonmerged
    section['depth'] = zint
    section['time'] = timemerged
    section[varname] = varmerged
    
    
    ## plot
    plt.rcParams.update({'font.size': 10,'xtick.labelsize': 10, 'ytick.labelsize': 10})

    if plot==1:
        plt.close('all')
        # double check profile averaging by running only first few iterations of loop
        plt.figure(figsize=(4,3))
        plt.plot(profdown_int,zint,label='down',marker='*')
        plt.plot(profup_int,zint,label='up')
        plt.plot(dummy,zint,label='mean')
        plt.legend()

        # double check positions
        plt.figure(figsize=(8,4))
        # plt.plot(ds['data'].lon,ds['data'].lat)
        plt.plot(lon,lat,marker='*',color='k',linestyle='')
        plt.plot(lonint,latint,marker='.',color='r',linestyle='')
        plt.plot(lonmerged,latmerged,marker='v',color='y',linestyle='')

    if varname=='salinity':
        ## plot to compare after inteprolatio
        fig,ax = plt.subplots(figsize=(8,4),ncols=2,sharey=True)
        vmin=34
        vmax=35.2
        cc=ax[0].scatter(time[ind0[i]:ind0[i+1]+1],zdown,c=profdown,s=10,vmin=vmin,vmax=vmax)
        ax[0].scatter(time[ind0[i+1]:ind0[i+2]],np.flip(zup),c=np.flip(profup),s=10,vmin=vmin,vmax=vmax)
        ax[1].scatter(np.ones(len(zint)),zint,c=profdown_int,vmin=vmin,vmax=vmax)
        ax[1].scatter(np.ones(len(zint))*2,zint,c=profup_int,vmin=vmin,vmax=vmax)
        ax[1].scatter(np.ones(len(zint))*3,zint,c=varmerged[-1,:],vmin=vmin,vmax=vmax)
        ax[0].invert_yaxis()
        plt.colorbar(cc,ax=ax[:])
        
    ## test indexing in loop below
    ind = 500
    fig,ax = plt.subplots(figsize=(8,3))
    plt.scatter(time[0:ind],z[0:ind],c=var[0:ind],s=10)
    plt.plot(time[list(np.array(ind0[0:30]))],z[list(np.array(ind0[0:30]))],marker='*',linestyle='None',color='r')
    ax.invert_yaxis()

    # output of function
    return section