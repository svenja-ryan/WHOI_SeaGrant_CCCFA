import xarray as xr
import numpy as np

#run_old = ['K365']
run_new = ['K003.hindcast','K004.thermhal90','K005.wind90']
xvel,yvel = 0,1

for var in ['none']:
    for i in np.arange(3):
        if var=='vel':
            data1 = xr.open_dataset('/home/sryan/clidex/data/ORCA/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '_1m_19580101_20161231_EIO_grid_U.nc')
        elif var=='TAUX':
            pathnew = '/climodes/data7/datasets/ORCA_JRA_tmp/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-'
            filenew = pathnew + run_new[i] + '_1957_2016_' + var + '_EIO_grid_U.nc'
            data1 = xr.open_dataset(filenew)
        elif var=='TAUY':
            pathnew = '/climodes/data7/datasets/ORCA_JRA_tmp/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-'
            filenew = pathnew + run_new[i] + '_1957_2016_' + var + '_EIO_grid_V.nc'
            data1 = xr.open_dataset(filenew)
        elif (var=='temp' or var =='sal' or var=='MLD'):
            pathnew = '/climodes/data7/datasets/ORCA_JRA_tmp/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-'
            filenew = pathnew + run_new[i] + '_1957_2016_' + var + '_EIO_grid_T.nc'
            data1 = xr.open_dataset(filenew)
        if 'data1' in globals():
            dsnew = data1.copy(deep=True)
            dsnew.coords['x']=dsnew['nav_lon'][0,:]
            dsnew.coords['y']=data1.nav_lat[:,0]
            dsnew = dsnew.rename({'x':'lon', 'y':'lat'}).set_coords(['lon','lat']).drop(['nav_lon','nav_lat'])
            #dsnew['vomecrty'].attrs['coordinates']=['lon','lat']
            del data1
            # save to netcdf
            dsnew.to_netcdf(filenew)
        
        #-----------------------------------------------------------------
        pathnew = '/climodes/data7/datasets/ORCA_JRA_tmp/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-'
        if xvel==1:
            data1 = xr.open_dataset('/home/sryan/clidex/data/ORCA/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '_1m_19580101_20161231_EIO_grid_U.nc')
            filenew = pathnew + run_new[i] + '_1957_2016_U_EIO_grid_U.nc'
        elif yvel==1:
            data1 = xr.open_dataset('/home/sryan/clidex/data/ORCA/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                              run_new[i] + '_1m_19580101_20161231_EIO_grid_V.nc')
            filenew = pathnew + run_new[i] + '_1957_2016_V_EIO_grid_V.nc'
        #
        #
        #print(data1)
        dsnew = data1.copy(deep=True)
        dsnew.coords['x']=dsnew['nav_lon'][0,:]
        dsnew.coords['y']=data1.nav_lat[:,0]
        dsnew = dsnew.rename({'x':'lon', 'y':'lat'}).set_coords(['lon','lat']).drop(['nav_lon','nav_lat'])
        #dsnew['vomecrty'].attrs['coordinates']=['lon','lat']
        del data1
        # save to netcdf
        dsnew.to_netcdf(filenew)
        #-----------------------------------------------------------------
        #pathold = '/data/sryan/ORCA/ORCA025-'
        #if (var=='temp'):
        #    key = 'votemper'
        #elif (var=='sal'):
        #    key = 'vosaline'
        #elif (var=='MLD'):
        #    key = 'somxl010'
        #elif (var=='vel'):
        #    key = 'vomecrty'
        #fileold = pathold + run_old[i] + '_1m_19480101_20071231_S30N_' + key + '_EIO.nc'
        #data2 = xr.open_dataset(fileold)
        #dsold = data2.copy(deep=True)
        #dsold.coords['x']=dsold['nav_lon'][0,:]
        #dsold.coords['y']=data2.nav_lat[:,0]
        #dsold = dsold.rename({'x':'lon', 'y':'lat'}).set_coords(['lon','lat']).drop(['nav_lon','nav_lat'])
        #dsold[key].attrs['coordinates']=['lon','lat']
        #del data2
        # save to netcdf
        #dsold.to_netcdf(fileold)