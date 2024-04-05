#!/usr/bin/env python
"""
SCHISM native reader
==================================
"""

import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
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
	filename = 'oceanum_ocean_nz_schism_v1_2D.nc',
	proj4 = proj_wgs84,
	use_3d = False,
	use_model_landmask = False)

# schism_native.plot_mesh(variable = ['sea_floor_depth_below_sea_level']) # check reader was correctly loaded

o.add_reader([reader_landmask,schism_native])
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""

###################################################################################
# Now compute LCS
###################################################################################

# start time for LCS computation
time_lcs_start  = [schism_native.start_time,schism_native.start_time +timedelta(hours=12.)] # can a single value or list of values
integration_time = timedelta(hours=24)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

# o = o.clone()
# Calculating attracting/backwards FTLE/LCS for integration time of T = 6 hours.
# built-in LCS computation
lcs = o.calculate_ftle(
    time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
    time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
    duration   = integration_time,    
    delta      = 0.05, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
    domain     = [171.0, 177.0, -39.0, -35.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
    ALCS       = True,  # attractive LCS, run backwards in time
    RLCS       = False) # repulsive LCS, run forward in time
# Convert LCS data to xarray 
import xarray as xr
data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
               'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),}  
ds_lcs = xr.Dataset(data_vars=data_dict, 
                coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})

# new green-cauchy tensors - still need to add squeezelines
lcs_new,ds_lcs_new = o.calculate_green_cauchy_tensor(
    time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
    time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
    duration   = integration_time,    
    delta      = 0.05, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
    domain     = [171.0, 177.0, -39.0, -35.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
    ALCS       = True,  # attractive LCS, run backwards in time
    RLCS       = False) # repulsive LCS, run forward in time

import pdb;pdb.set_trace()

import matplotlib.pyplot as plt;plt.ion();plt.show()
# example plots with xarray 
fig, ax = plt.subplots()
ds_lcs.ALCS.isel(time=0).plot(vmin=1e-7,vmax=1e-6)

fig, ax = plt.subplots()
ds_lcs_new.ALCS.isel(time=0).plot(vmin=8e-6,vmax=9e-6)

# >> patterns are the same, but LCS magnitude range is quite different 

# >> next step is to compute squeezelines, uisng saved variables C11,C12,C22
# see https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L12