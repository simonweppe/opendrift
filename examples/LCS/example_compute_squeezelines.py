import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift

# Try to compute squeezelines from the ds_lcs_new
import sys
sys.path.append('/home/simon/code/github/opendrift_simon/examples/LCS')
from cLCS_tools import compute_cLCS_squeezelines
import xarray as xr

###############################
# MODEL
###############################
if False:
    o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information
    reader_landmask = reader_global_landmask.Reader()
    proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

    schism_native = reader_schism_native.Reader(
        filename = '/home/simon/calypso_science/projects/NP_disposalgrounds/opendrift_modelling/oceanum_ocean_nz_schism_v1_2D.nc',
        proj4 = proj_wgs84,
        use_3d = False,
        use_model_landmask = False)

    cst_reader_wind = reader_constant.Reader( {'x_wind': 0, 'y_wind': 0,}) # add a constant reader just to check it's used in the LCS computations >> OK

    o.add_reader([reader_landmask,schism_native,cst_reader_wind])
    o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
    o.set_config('general:coastline_action', 'previous') # prevent particles stranding, free-slip boundary
    o.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme
    o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""

    # start time for LCS computation
    time_lcs_start  = [schism_native.start_time,schism_native.start_time +timedelta(hours=12.)] # can a single value or list of values
    integration_time = timedelta(hours=12)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

    # # new green-cauchy tensors - still need to add squeezelines
    lcs_new,ds_lcs = o.calculate_green_cauchy_tensor(
        reader     = schism_native,
        time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
        time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
        duration   = integration_time,    
        delta      = 3000, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
        domain     = [170.0, 174.0, -40.0, -38.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
        ALCS       = True,  # attractive LCS, run backwards in time
        RLCS       = True, # repulsive LCS, run forward in time
        cartesian_epsg = 2193)
    
    # new green-cauchy tensors - still need to add squeezelines
    lcs = o.calculate_ftle(
        reader     = schism_native,
        time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
        time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
        duration   = integration_time,    
        delta      = 3000, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
        domain     = [170.0, 174.0, -40.0, -38.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
        ALCS       = True,  # attractive LCS, run backwards in time
        RLCS       = True, # repulsive LCS, run forward in time
        cartesian_epsg = 2193)    
    # Convert LCS data to xarray 
    import xarray as xr
    data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
                'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),}  
    ds_lcs1 = xr.Dataset(data_vars=data_dict, 
                    coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})


    #################################################################
    # quick check plots of FTLE
    #################################################################
    import matplotlib.pyplot as plt;plt.ion();plt.show()
    fig, ax = plt.subplots(2,2)
    np.abs(ds_lcs1).ALCS.isel(time=0).plot(ax=ax[0,0],vmin=1e-5,vmax=1.5e-4)
    ax[0,0].set_title('attractive LCS - Built-in Method ')
    np.abs(ds_lcs).ALCS.isel(time=0).plot(ax=ax[0,1],vmin=1e-5,vmax=1.5e-4)
    ax[0,1].set_title('attractive LCS - Duran ')

    np.abs(ds_lcs1).RLCS.isel(time=0).plot(ax=ax[1,0],vmin=1e-5,vmax=1.5e-4)
    ax[1,0].set_title('repuslive LCS - Built-in Method ')
    np.abs(ds_lcs).RLCS.isel(time=0).plot(ax=ax[1,1],vmin=1e-5,vmax=1.5e-4)
    ax[1,1].set_title('repuslive LCS - Duran ')


#################################################################
# Compute "squeezelines"
#################################################################
# see script to compute squeezeline here:
# https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L12

ds_lcs = xr.open_dataset('ds_lcs_new.nc')
obj=compute_cLCS_squeezelines(ds_lcs,arclength = 20000)
obj.run()
# squeezelines are saved here obj.pxt, obj.pyt, obj.pzt
import matplotlib.pyplot as plt;plt.ion();plt.show()
fig, ax = plt.subplots(1,1)
ax.pcolormesh(ds_lcs['X'],ds_lcs['Y'],np.abs(ds_lcs).ALCS.isel(time=0))
[ax.plot(x,y,'grey') for x,y in zip(obj.pxt,obj.pyt) ]
ax.set_aspect('equal')
import pdb;pdb.set_trace()

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