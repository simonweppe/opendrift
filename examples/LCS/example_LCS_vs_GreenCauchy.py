#!/usr/bin/env python
"""
LCS from SCHISM NZ domain

Compare LCS metrics obtained with opendrift built-in methods, with methods adapted from cLCS (Duran, Montano)

----
Ongoing work to add methods from Mireya's/Duran's cLCS toolbox

- calculate_Cauchy_Green() : DONE
    computation of Green-Cauchy tensor (incl. C11,C12,C22) https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L352

- computation of squeezeline from C11,C12,C22 : TO DO 
    https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L43
    see the run() function of that class

- See full cLCS example with squeezeline comoutations
  here : https://github.com/MireyaMMO/cLCS/blob/main/examples/01_cLCS_ROMS.ipynb

"""

import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift

###############################
# MODEL
###############################
o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information
###############################
# READERS
###############################
# Creating and adding reader using a native SCHISM netcdf output file
# SCHISM reader
reader_landmask = reader_global_landmask.Reader()

# NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
# proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

schism_native = reader_schism_native.Reader(
	filename = '/home/simon/calypso_science/projects/NP_disposalgrounds/opendrift_modelling/oceanum_ocean_nz_schism_v1_2D.nc',
	proj4 = proj_wgs84,
	use_3d = False,
	use_model_landmask = False)
# schism_native.plot_mesh(variable = ['sea_floor_depth_below_sea_level']) # check reader was correctly loaded

cst_reader_wind = reader_constant.Reader( {'x_wind': 0, 'y_wind': 0,}) # add a constant reader just to check it's used in the LCS computations >> OK

o.add_reader([reader_landmask,schism_native,cst_reader_wind])
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""
o.set_config('general:coastline_action', 'previous') # prevent particles stranding, free-slip boundary
o.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme

###################################################################################
# Now compute LCS
###################################################################################

# start time for LCS computation
time_lcs_start  = [schism_native.start_time,schism_native.start_time +timedelta(hours=12.)] # can a single value or list of values
integration_time = timedelta(hours=12)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

# o = o.clone()
# Calculating attracting/backwards FTLE/LCS for integration time of T = 6 hours.
# built-in LCS computation
if True:
    lcs = o.calculate_ftle(
        reader     = schism_native, # reader used to define the seeding frame. Needs to have correct proj4 defined. If not specified it will use the first reader defined in o.add_reader()
        time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
        time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
        duration   = integration_time,    
        delta      = 10000, # spatial step in meters
        domain     = [171.0, 175.0, -40.0, -38.0], # user-defined frame within reader domain, in native coordinates [xmin, xmax, ymin, ymax], if None use entire domain. If lon,lat it will be converted to <cartesian_epsg> first
        ALCS       = True,  # attractive LCS, run backwards in time
        RLCS       = False, # repulsive LCS, run forward in time
        cartesian_epsg = 2193 ) # the reader has native lon/lat so we need to specify a cartesian coordinate system
    
    # Convert LCS data to xarray 
    import xarray as xr
    data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
                'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),}  
    ds_lcs = xr.Dataset(data_vars=data_dict, 
                    coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})
    import pdb;pdb.set_trace()
# new green-cauchy tensors - still need to add squeezelines
lcs_new,ds_lcs_new = o.calculate_green_cauchy_tensor(
    reader     = schism_native,
    time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
    time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
    duration   = integration_time,    
    delta      = 10000, # spatial step (in meters)
    domain     = [171.0, 175.0, -40.0, -38.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
    ALCS       = True,  # attractive LCS, run backwards in time
    RLCS       = False, # repulsive LCS, run forward in time
    cartesian_epsg = 2193)

# note the ds_lcs_new xarray object can easily be saved to netcdf using ds_lcs_new.to_netcdf()

# compare FTLE from built-in opendrift function, and FTLE from cLCS toolbox
# 
import matplotlib.pyplot as plt;plt.ion();plt.show()
# example plots with xarray 
fig, ax = plt.subplots()
np.abs(ds_lcs).ALCS.isel(time=0).plot(vmin=1e-5,vmax=1.5e-4)
ax.set_title('Built-in Method for LCS')

fig, ax = plt.subplots()
np.abs(ds_lcs_new).ALCS.isel(time=0).plot(vmin=1e-5,vmax=1.5e-4)
ax.set_title('Duran methods for LCS')

import pdb;pdb.set_trace()


# >> patterns are the same, but LCS magnitude range is quite different 
# 
#  after using final position rather than net displacement (as in ftle), we get more consistent FLTE magnitudes
# 