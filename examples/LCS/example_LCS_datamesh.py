#!/usr/bin/env python
"""
LCS from Datamesh tidal flows (regular grid)
==================================
"""

from datetime import datetime, timedelta
from opendrift.readers import reader_datamesh_regular_cons,reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
import numpy as np


o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
# reader_landmask = reader_global_landmask.Reader()  # not needed, we use the depth info as land_binary_mask
datamesh_regular_cons = reader_datamesh_regular_cons.Reader(
      filename = '/home/simon/calypso_science/tide_grids/oceanum_2km.zarr',)  # native coordinate system is lon/lat

o.add_reader([datamesh_regular_cons]) #
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding
o.set_config('drift:horizontal_diffusivity', 0.0) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)

# Now compute LCS

# start time for LCS computation
time_lcs_start  = [datetime(2024,1,1),datetime(2024,1,1) +timedelta(hours=12.)] # can a single value or list of values
integration_time = timedelta(hours=6)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

# o = o.clone()
# Calculating attracting/backwards FTLE/LCS for integration time of T = 6 hours.

lcs = o.calculate_ftle(
    time       = time_lcs_start, # the start time of LCS computation ..can be a single value or list of values
    time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
    duration   = integration_time,    
    delta      = 0.02, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
    domain     = [173.0, 176., -42.0, -39.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
    ALCS       = True,  # attractive LCS, run backwards in time
    RLCS       = False) # repulsive LCS, run forward in time

import pdb;pdb.set_trace()
 
# Convert LCS data to xarray 
import xarray as xr
data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
               'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),}  
ds = xr.Dataset(data_vars=data_dict, 
                coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})
ds.ALCS.isel(time=0).plot(vmin=1e-7,vmax=1e-5)

#plot mean of LCS over time
ds.ALCS.mean(dim= 'time').plot(vmin=1e-7,vmax=1e-5)

########################################################################3
# add a cartopy map and plot LCS
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
plt.ion()
plt.show()
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}) #central_longitude = 180 , center map around dateline (typical for NZ projects) (default is 0)
ax.set_extent([lcs['lon'].min(),lcs['lon'].max(),lcs['lat'].min(),lcs['lat'].max()], crs=ccrs.PlateCarree()) # NZ
ax.set_facecolor([ 0.59375 , 0.71484375, 0.8828125 ]) # cartopy blue
ax.gridlines(draw_labels=True)
ax.pcolormesh(lcs['lon'],lcs['lat'],lcs['ALCS'].squeeze()[0,:,:],vmin=1e-7,vmax=1e-5,transform=ccrs.PlateCarree(),zorder = 0)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='intermediate', levels=None),facecolor=cfeature.COLORS['land'])

import pdb;pdb.set_trace()