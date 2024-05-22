import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift
import pandas as pd

# Try to compute squeezelines from the ds_lcs_new
import sys
sys.path.append('/home/simon/code/github/opendrift_simon/examples/LCS')
import xarray as xr

###############################
# MODEL
###############################

o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
reader_landmask = reader_global_landmask.Reader()
proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

schism_native = reader_schism_native.Reader(
    filename = '/home/simon/calypso_science/projects/XXXX_Gigablue/opendrift_modelling/nz_schism_v1_2D_subset_*.nc',
    proj4 = proj_wgs84,
    use_3d = False,
    use_model_landmask = False)

cst_reader_wind = reader_constant.Reader( {'x_wind': 0, 'y_wind': 0,}) # add a constant reader just to check it's used in the LCS computations >> OK

o.add_reader([reader_landmask,schism_native,cst_reader_wind])
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding, free-slip boundary
o.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme
o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""

# LCS OPTIONS
# FRAME = [173.0,-43.0,179.0,-41.0]
FRAME = [173.5,178.,-42.95,-41.05]
DX = 1000
# release every day at midday , and run for 7 days
time_lcs_start  = np.arange(schism_native.start_time,schism_native.start_time+timedelta(days=30.) ,timedelta(hours=12.)) # can a single value or list of values
time_lcs_start = pd.to_datetime(time_lcs_start).to_pydatetime() # make sure it's python-datetime format
integration_time = timedelta(days=7.0)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

if False:

    for start_time in time_lcs_start[29:]:
        # Compute green-cauchy tensors with new method
        lcs_new,ds_lcs = o.calculate_green_cauchy_tensor(
            reader     = schism_native,
            time       = start_time, # the start time of LCS computation ..can be a single value or list of values
            time_step  = timedelta(minutes=30), # time step of individual opendrift simulations
            duration   = integration_time,    
            delta      = DX, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
            domain     = FRAME, # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
            ALCS       = True,  # attractive LCS, run backwards in time
            RLCS       = True, # repulsive LCS, run forward in time
            cartesian_epsg = 2193)
        ds_lcs.to_netcdf('./outputs/lcs_%s.nc' % start_time.strftime('%Y%m%d_%H%M'))

        # compute FTLE with built-in method for comparison
        # lcs = o.calculate_ftle(
        #     reader     = schism_native,
        #     time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
        #     time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
        #     duration   = integration_time,    
        #     delta      = 3000, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
        #     domain     = [170.0, 174.0, -40.0, -38.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
        #     ALCS       = True,  # attractive LCS, run backwards in time
        #     RLCS       = True, # repulsive LCS, run forward in time
        #     cartesian_epsg = 2193)    
        
        # Convert LCS data to xarray 
        # import xarray as xr
        # data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
        #             'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),}  
        # ds_lcs1 = xr.Dataset(data_vars=data_dict, 
        #                 coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})

    import pdb;pdb.set_trace()



    #################################################################
    # quick check plots of FTLE
    #################################################################

    fig, ax = plt.subplots()
    (ds_lcs).ALCS.isel(time=0).plot()#vmin=1e-5,vmax=1.5e-4)

    import matplotlib.pyplot as plt;plt.ion();plt.show()
    fig, ax = plt.subplots(2,2)
    np.abs(ds_lcs).ALCS.isel(time=0).plot(ax=ax[0,0],vmin=1e-5,vmax=1.5e-4)
    ax[0,0].set_title('attractive LCS - Built-in Method ')
    np.abs(ds_lcs).ALCS.isel(time=0).plot(ax=ax[0,1],vmin=1e-5,vmax=1.5e-4)
    ax[0,1].set_title('attractive LCS - Duran ')

    np.abs(ds_lcs).RLCS.isel(time=0).plot(ax=ax[1,0],vmin=1e-5,vmax=1.5e-4)
    ax[1,0].set_title('repuslive LCS - Built-in Method ')
    np.abs(ds_lcs).RLCS.isel(time=0).plot(ax=ax[1,1],vmin=1e-5,vmax=1.5e-4)
    ax[1,1].set_title('repuslive LCS - Duran ')

if True:
#################################################################
# Compute Cauchy–Green strain tensorlines aka "squeezelines"
#################################################################
# see script to compute squeezeline here:
# https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L12
    import matplotlib.pyplot as plt;plt.ion();plt.show()
    from cLCS_tools import compute_cLCS_squeezelines,get_colourmap,plot_colourline

    ds_lcs1= xr.open_mfdataset('./outputs/lcs_*.nc')
    # apply mask 
    mask = np.isnan(ds_lcs1.isel(time=28).ALCS)
    ds_lcs = ds_lcs1.where(~mask,0.0) # mask with 0.0 rather than Nans
    # keep orginal X,Y
    ds_lcs['X'] = ds_lcs1['X']
    ds_lcs['Y'] = ds_lcs1['Y'] 
    ds_lcs_coarsened = ds_lcs.coarsen(lon=4, lat=4).mean() # coarsen LCS field prior to computing squeezelines
    # nxb,nyb governs the density of computed CG tensorlines
    # arclength in meters. i.e. the number of segments of each line
    obj=compute_cLCS_squeezelines(ds_lcs_coarsened, nxb=30, nyb=30, arclength = 30000)                        
    obj.run()
    import pdb;pdb.set_trace()

    # squeezelines are saved here obj.pxt, obj.pyt, obj.pzt
    fig, ax = plt.subplots(1,1)
    ax.pcolormesh(ds_lcs['X'].isel(time=0),ds_lcs['Y'].isel(time=0),np.abs(ds_lcs).ALCS.mean(dim='time'))
    # [ax.plot(x,y,'grey') for x,y in zip(obj.pxt,obj.pyt) ]
    # only plot line for which pzt is not nan
    [ax.plot(x,y,'grey') for x,y,z in zip(obj.pxt,obj.pyt,obj.pzt)  if ~np.isnan(z).any() ]
    ax.set_aspect('equal')
    ax.set_title('ALCS')
    
    ########################################################
    # make nicer plots - ALCS
    ########################################################
    import sys
    sys.path.append('/home/simon/code/github/toolbox_simon/calypso_tools/')
    from toolbox import add_basemap,rotate_vector
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    data = ds_lcs.ALCS.mean(dim='time').where(~mask,np.nan)
    srf = ax.pcolormesh(ds_lcs['X'].isel(time=0),ds_lcs['Y'].isel(time=0),data,cmap=get_colourmap('Duran_cLCS'))
    # coarsen_fac = 2
    # data = ds_lcs.ALCS.where(~mask,np.nan).mean(dim='time').coarsen(lon=coarsen_fac, lat=coarsen_fac).mean()
    srf = ax.pcolormesh(ds_lcs['X'].isel(time=0).coarsen(lon=coarsen_fac, lat=coarsen_fac).mean(),ds_lcs['Y'].isel(time=0).coarsen(lon=coarsen_fac, lat=coarsen_fac).mean(),data,cmap=get_colourmap('Duran_cLCS'))
    cbar = fig.colorbar(srf,label='Attractive LCS')
    add_basemap(ax,crs_code=2193,zoom_level=8)
    ax.set_aspect('equal')
    ax.set_xlim([1.67e6,1.90e6])
    ax.set_ylim([5.27e6,5.44e6])
    fig.savefig('attractive_lcs.png',bbox_inches='tight')
    [ax.plot(x,y,'grey') for x,y in zip(obj.pxt,obj.pyt) ]
    fig.savefig('attractive_lcs+squeezeline1.png',bbox_inches='tight')

    ########################################################
    # make nicer plots - RLCS
    ########################################################
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    data = ds_lcs.RLCS.mean(dim='time').where(~mask,np.nan)
    srf = ax.pcolormesh(ds_lcs['X'].isel(time=0),ds_lcs['Y'].isel(time=0),data,cmap=get_colourmap('Duran_cLCS'))
    cbar = fig.colorbar(srf,label='Repulsive LCS')
    ax.set_aspect('equal')
    add_basemap(ax,crs_code=2193,zoom_level=8)
    ax.set_xlim([1.67e6,1.90e6])
    ax.set_ylim([5.27e6,5.44e6])
    fig.savefig('repulsive_lcs.png',bbox_inches='tight')

    ########################################################
    # now look at gradients
    ########################################################
    fig, ax = plt.subplots(1,1,figsize=(16,9))
    data = ds_lcs.ALCS.mean(dim='time').where(~mask,np.nan)
    srf = ax.pcolormesh(ds_lcs['X'].isel(time=0),ds_lcs['Y'].isel(time=0),data,cmap=get_colourmap('Duran_cLCS'))
    # Note : 
    # np.gradient(data.values, axis=1) is gradient along column aka X-dimension
    # np.gradient(data.values, axis=0) is gradient along line aka Y-dimension
    U = np.gradient(data.values, axis=1)
    V = np.gradient(data.values, axis=0)
    ax.quiver(ds_lcs['X'].isel(time=0),ds_lcs['Y'].isel(time=0), U,V)
    ax.streamplot(ds_lcs['X'].isel(time=0).values,ds_lcs['Y'].isel(time=0).values, U,V,density=0.6, color='k')
    ########################################################

if False:
    # plot as coloured lines
    from cLCS_tools import get_colourmap,plot_colourline
    fig, ax = plt.subplots(1,1)
    cmap = get_colourmap('Duran_cLCS')
    # Plot all >> too heavy
    # [plot_colourline(x,y,z,cmap,ax=ax) for x,y,z in zip(obj.pxt,obj.pyt,obj.pzt) ]
    # plot one
    line_id = 10
    x,y,z = obj.pxt[line_id],obj.pyt[line_id],obj.pzt[line_id]
    z=np.abs(z)
    ax.plot(x,y)
    plot_colourline(x,y,z,cmap,ax=ax) 