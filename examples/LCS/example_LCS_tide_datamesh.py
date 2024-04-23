import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_datamesh_schism_cons
from opendrift.readers import reader_global_landmask
from opendrift.readers import reader_shape
from opendrift.models.oceandrift import OceanDrift
import pandas as pd
import xarray as xr
import sys
sys.path.append('/home/simon/code/github/toolbox_simon/calypso_tools/')
from toolbox import find_closest_node,add_basemap,convert_coords
from scipy.signal import find_peaks
import oceantide

###############################
# MODEL
###############################
o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
###############################
# READERS
###############################

schism_datamesh_cons = reader_datamesh_schism_cons.Reader(
	filename = '/home/simon/calypso_science/tide_grids/calypso-tidalcons-hauraki-v1',)  # native coordinate system is lon/lat
# this will include the mesh boundary polygons and interior islands
o.add_reader([schism_datamesh_cons]) #
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding
o.set_config('drift:horizontal_diffusivity', 0.0) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)
o.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme
o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""
###############################
# SEEDING
# 
# Find incoming/outgoing tides
###############################
#
tide_cons=schism_datamesh_cons.dataset.copy()
tide_cons['con']=[x.strip().upper() for x in tide_cons['cons'].values]
#reference location for the tidal LCS
X=174.818172
Y=-36.835787
timearray = np.arange(datetime(2024,1,1), datetime(2024,2,1), timedelta(minutes=30)).astype(datetime)
dd,node_id = find_closest_node(tide_cons.lon,tide_cons.lat,X,Y)
tide_ts_node  = tide_cons.isel(node=node_id).tide.predict(times=timearray)#, time_chunk=50, components=["h", "u", "v"])
# import matplotlib.pyplot as plt;plt.ion();plt.show()
# tide_ts_node.h.plot()
# find high and low tide
ht=find_peaks(tide_ts_node.h.squeeze())[0]
lt=find_peaks(-tide_ts_node.h.squeeze())[0]
lt=lt[lt>ht[0]] # get rid of first low tide if it is before first high tide

# LCS OPTIONS
FRAME = [174.7,174.9,-36.86,-36.77]
DX = 200  # 

# # simple seeding to test run
# o.seed_elements(lon=X,
#                 lat=Y, 
#                 radius=1000, # 1km radius  > this is to test the closest_ocean_point() method
#                 number=1000,
#                 z=np.linspace(0,-10, 1000), 
#                 time=timearray[0]) # this will be a continuous release over that time vector
# # Running model
# o.run(time_step=900,
# 	  end_time = timearray[48])
# 	  # outfile='schism_native_output.nc')
# o.plot(fast=True)
# o.animation(fast=True)

# now run for each stage hour of the tide
for ii in range(0,len(ht)-1):
    integration_time = timedelta(hours=1.0)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

    # high to low tide
    time_lcs_start = timearray[ht[ii]:lt[ii]] # incoming tide, start each new hour after high tide
    for start_time in time_lcs_start:
        # Compute green-cauchy tensors with new method
        lcs_new,ds_lcs = o.calculate_green_cauchy_tensor(
            reader     = schism_datamesh_cons,
            time       = start_time, # the start time of LCS computation ..can be a single value or list of values
            time_step  = timedelta(minutes=10), # time step of individual opendrift simulations
            duration   = integration_time,    
            delta      = DX, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
            domain     = FRAME, # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
            ALCS       = True,  # attractive LCS, run backwards in time
            RLCS       = True, # repulsive LCS, run forward in time
            cartesian_epsg = 2193)
        ds_lcs.to_netcdf('./outputs_lcs_tide/lcs_%s.nc' % start_time.strftime('%Y%m%d_%H%M'))        
    
    # low to high tide
    time_lcs_start = timearray[lt[ii]:ht[ii+1]] # outgoing tide tide
    for start_time in time_lcs_start:
        # Compute green-cauchy tensors with new method
        lcs_new,ds_lcs = o.calculate_green_cauchy_tensor(
            reader     = schism_datamesh_cons,
            time       = start_time, # the start time of LCS computation ..can be a single value or list of values
            time_step  = timedelta(minutes=10), # time step of individual opendrift simulations
            duration   = integration_time,    
            delta      = DX, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
            domain     = FRAME, # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
            ALCS       = True,  # attractive LCS, run backwards in time
            RLCS       = True, # repulsive LCS, run forward in time
            cartesian_epsg = 2193)
        ds_lcs.to_netcdf('./outputs_lcs_tide/lcs_%s.nc' % start_time.strftime('%Y%m%d_%H%M'))     





#     import pdb;pdb.set_trace()
    
# import pdb;pdb.set_trace()
# o.plot(fast=True,filename='test_schism_datamesh_cons.png')
# o.animation(fast=True,filename='test_schism_datamesh_cons.gif')

# o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
# reader_landmask = reader_global_landmask.Reader()
# proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

# schism_native = reader_schism_native.Reader(
#     filename = '/home/simon/calypso_science/projects/XXXX_Gigablue/opendrift_modelling/nz_schism_v1_2D_subset_*.nc',
#     proj4 = proj_wgs84,
#     use_3d = False,
#     use_model_landmask = False)

# cst_reader_wind = reader_constant.Reader( {'x_wind': 0, 'y_wind': 0,}) # add a constant reader just to check it's used in the LCS computations >> OK

# o.add_reader([reader_landmask,schism_native,cst_reader_wind])
# o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
# o.set_config('general:coastline_action', 'previous') # prevent particles stranding, free-slip boundary




