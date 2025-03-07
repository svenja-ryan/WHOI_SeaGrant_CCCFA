import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xarray as xr
import cartopy.crs as ccrs
import cartopy
import datetime
import xroms
import warnings
import cmocean as cmo
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")  # Ignores all warnings

# load bathymetry data
bathy = xr.open_dataset('/vast/clidex/data/bathymetry/ETOPO1/ETOPO1_Bed_g_gmt4.grd').sel(x=slice(-85,-40),y=slice(25,55)).load()

#############################################################
############### Mapping functions ###########################
#############################################################

## plot manual clabels
def manual_clabel(x,y,ax,cc,proj):
    for ind in range(len(x)):
        xy_trans = proj.transform_point(x[ind],y[ind],ccrs.PlateCarree())
        manual_location = [(xy_trans[0],xy_trans[1])]
        ax.clabel(cc,fontsize=8,fmt='%1d',inline=1, manual=manual_location)
#        
#
#
## projection for Northwest Atlantic rotated
proj = ccrs.LambertConformal(central_latitude = 40, 
                              central_longitude = -70, 
                              standard_parallels = (25, 25))
#
#
#
### function to plot full map
def plot_map(ax,extent,c=np.arange(0,200,10),plotbathy=None,inc=0.5):
    """
    Produces map with coastlines using cartopy.
    
    INPUT:
    ax       : axis handle 
    extent   : [lon1,lon2,lat1,lat2]
    
    OUTPUT:
    gl       : grid handle
    
    """
    
    ## plot bathymetry
    if plotbathy:
        cc=ax.contourf(bathy.x,bathy.y,bathy.z*(-1),levels=c,cmap='Blues',transform=ccrs.PlateCarree(),vmax=c[-1],vmin=c[0],extend='both')
        ll = ax.contour(bathy.x,bathy.y,bathy.z*(-1),levels=[10,30,60,100],colors='k',transform=ccrs.PlateCarree(),linewidths=0.5,alpha=0.5)
        ax.contour(bathy.x,bathy.y,bathy.z*(-1),levels=[60],colors='r',transform=ccrs.PlateCarree(),linewidths=1,alpha=0.5)
        # plt.clabel(ll,levels=[60,100,200],fmt='%1dm')
        plt.colorbar(cc,shrink=0.6,label='depth [m]')
        # manual_locations = [(-69.8, 41.5), (-69.5, 41.5), (-69.6, 41.7)]
        x = [-69.8,-69.5,-69.6,-69.5]
        y = [41.5,41.5,41.7,41.8]
        manual_clabel(x,y,ax,ll,proj)

    ## formatting
    ax.set_extent(extent)
#     ax.coastlines(resolution='50m',color='gray')
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linestyle='-',alpha=0.3,
                      xlocs=np.arange(-80,-60,inc),ylocs=np.arange(30,50,inc),x_inline=False)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='high'), facecolor='lightgray',edgecolor='None')
    gl.rotate_labels = False
    gl.right_labels = False
    gl.top_labels = False

    return gl
#
#
#
def datetime2matlab(time):
    import datetime
    """
    INPUT
    time: timevectors as datetime64 fromat
    
    OUTPUT
    timenum: time in Matlab's datenum format
    """
    timenum = mdates.date2num(time)+datetime.date(1971, 1, 2).toordinal()
    return timenum
#
#
#
#---------------------------------------------------------
# plot rectangles for 6 regions
#---------------------------------------------------------

# create boxes around the original points
from geopy.distance import great_circle

# Function to calculate a point given a distance and bearing from a starting point
def point_at_distance(start, distance, bearing):
    """
    Calculate a point given a start point, distance, and bearing.
    """
    origin = (start[0], start[1])
    destination = great_circle(kilometers=distance*1.89).destination(origin, bearing) # convert nautical miles to km
    return (destination.latitude, destination.longitude)

def rectangle_around_point(point,dx,dy,angle=0):
    # Distance in nautical miles
    distance_north_south_nm = dy
    distance_east_west_nm = dx
    
    # Define the bearings for the four corners of the rectangle
    bearings = [0, 90, 180, 270]
    if angle==0:
        bearings = bearings
    else:
        bearings = np.array(bearings)-angle
        bearings[0] = np.array(bearings[0])+360
    
    # Calculate the four corners of the rectangle
    north = point_at_distance(point, distance_north_south_nm, bearings[0])  # North
    south = point_at_distance(point, distance_north_south_nm, bearings[2])  # South
    east = point_at_distance(point, distance_east_west_nm, bearings[1])  # East
    west = point_at_distance(point, distance_east_west_nm, bearings[3])  # West
    
    # Define corners based on combinations of north/south and east/west
    corners = [
        point_at_distance(north, distance_east_west_nm, bearings[1]),  # NE
        point_at_distance(north, distance_east_west_nm, bearings[3]),  # NW
        point_at_distance(south, distance_east_west_nm, bearings[3]),  # SW
        point_at_distance(south, distance_east_west_nm, bearings[1])  # SE
    ]
    
    # Repeat the last point to close the rectangle
    corners.append(corners[0])
    lats, lons = zip(*corners)
    return lats, lons

# function to plot all
def plot_regions(ax):

    # stations we agree on
    rs1 = [41.523666, -69.466644]  # RS1 - southern most, Josiah summer
    rs2 = [41.606559, -69.762969]  # RS2 - off Chatham, summer skate
    rs3 = [41.825530, -69.856974]  # RS3 - off Nauset Harbor (Kurt/Eric summer)
    rs4 = [41.997377, -69.907179]  # RS4 - NE Nauset Beach - Sean fall/winter
    rs5 = [42.157877, -70.054506]  # RS5 - NE Provincetown - GOM Survey Oct-Nov, April-May
    rs6 = [42.006187, -70.211246]  # RS6 - CCB, Ptown Buoy - Braden

    # add station locations
    for ds,dx,dy,orientation in zip([rs3,rs4,rs5,rs6],
                                    [1,1,1,1,1],
                                    [2,2,2,1],
                                    [10,25,25,0]):
        # ax.plot(ds[1],ds[0],marker='s',color='r',markersize=2,transform=ccrs.PlateCarree())
        # Plot the rectangle
        lats,lons = rectangle_around_point(ds,dx,dy,angle=orientation)
        # print(np.round(lats,2))
        # print(np.round(lons,2))
        ax.plot(lons, lats, 'r',transform=ccrs.PlateCarree())
    
    # add parallelograpm
    lon = [-69.8,-69.5,-69.3,-69.65,-69.8]
    lat = [41.65,41.65,41.45,41.4,41.65]
    ax.plot(lon, lat, 'r',transform=ccrs.PlateCarree())





#############################################################
############### General stuff ###########################
#############################################################
def font_medium():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 10
    #csfont = {'fontname':'Comic Sans MS'}
    legend_properties = {'weight':'bold'}
    #font.family: sans-serif
    #font.sans-serif: Helvetica Neue

    #import matplotlib.font_manager as font_manager
    #font_dirs = ['/home/mhell/HelveticaNeue/', ]
    #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    #font_list = font_manager.createFontList(font_files)
    #font_manager.fontManager.ttflist.extend(font_list)

    plt.rc('font', size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')          # controls default text sizes
    #plt.rc('font', size=SMALL_SIZE, serif='DejaVu Sans', weight='light')
    plt.rc('text', usetex='false')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelweight='normal')     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight='normal') #, family='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=False) #,



def font_for_print():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10
    #csfont = {'fontname':'Comic Sans MS'}
    legend_properties = {'weight':'bold'}
    #font.family: sans-serif
    #font.sans-serif: Helvetica Neue

    #import matplotlib.font_manager as font_manager
    #font_dirs = ['/home/mhell/HelveticaNeue/', ]
    #font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    #font_list = font_manager.createFontList(font_files)
    #font_manager.fontManager.ttflist.extend(font_list)

    plt.rc('font', size=SMALL_SIZE, serif='Helvetica Neue', weight='normal')          # controls default text sizes
    #plt.rc('font', size=SMALL_SIZE, serif='DejaVu Sans', weight='light')
    plt.rc('text', usetex='false')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelweight='normal')     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE, labelweight='normal') #, family='bold')    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE, titleweight='bold', autolayout=False) #, family='bold')  # fontsize of the figure title