#!/usr/bin/env python
"""
====================================================================
Prototype for surface phase
====================================================================
"""
import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_shape
try:    
    from opendrift.readers import reader_datamesh_regular_cons
    from opendrift.readers import reader_datamesh_schism_cons
except:
    import sys
    sys.path.append('/home/simon/code/github/opendrift_simon/opendrift/readers')
    import reader_datamesh_schism_cons

import geopandas as gpd
# import matplotlib.pyplot as plt;plt.ion();plt.show()
import xarray as xr

###############################
# READERS
###############################

# use built-in landmask - consider using shape file if necessary
# reader_landmask = reader_global_landmask.Reader()

reader_landmask = reader_shape.Reader.from_shpfiles('./shp_files/osm_shoreline_wgs84.shp')
#  MUST BE WGS84, or need to specify projection



import xwrf
ds = xr.open_dataset('wrfout_2025_07_08_d03.nc').xwrf.postprocess()
# we need to get rid of the x_stag,y_stag variables as they will confuse the reader otherwise 
# as they have the same standard_name = projection_x_coordinate
# ds = ds.drop_vars(['x_stag', 'y_stag']).copy(deep=True)     # we could also just remove x_stag,y_stag that confuses the reader 
# or we simply only keep what we need
ds_sub = ds[['wind_east_10','wind_north_10','wrf_projection']].copy(deep=True) 
proj4_str = str(ds.wrf_projection.data) # projection to fall back on WGS84 data
# Atmospheric model for wind
reader_wind_wrf = reader_netCDF_CF_generic.Reader(      filename = ds_sub, # input the processed xwrf dataset
                                                        standard_name_mapping={ 'wind_east_10':'x_wind',
                                                                                'wind_north_10':'y_wind'},
                                                        proj4 = proj4_str)

reader_wind_era = reader_netCDF_CF_generic.Reader(      filename = 'wind_era5_wind10m_2022-07-08 00:00:00.nc', # input the processed xwrf dataset
                                                        standard_name_mapping={ 'u10':'x_wind',
                                                                                'v10':'y_wind'},)                                                   

# reader_landmask = reader_global_landmask.Reader() 
schism_datamesh_cons = reader_datamesh_schism_cons.Reader(
	filename = '/home/simon/calypso_science/tide_grids/calypso-tidalcons-brest-v3.zarr',)  # native coordinate system is lon/lat
# this will include the mesh boundary polygons and interior islands

###############################
# MODEL
###############################
o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information

# o.add_reader([reader_landmask,reader_bathy,reader_current,reader_wind_era5])
o.add_reader([reader_landmask,
            #   reader_wind_era,
              reader_wind_wrf,
              schism_datamesh_cons])


o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding
o.set_config('general:seafloor_action','deactivate')
o.set_config('drift:horizontal_diffusivity', 0.1) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)
o.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme
o.set_config('drift:vertical_advection', False)
o.set_config('drift:vertical_mixing', False) # dont use for now, only buoyancy effects
o.set_config('vertical_mixing:timestep',60) # sub time step for vertical mixing
o.set_config('vertical_mixing:diffusivitymodel','environment') # use environment data, submodels can be specified
o.set_config('environment:fallback:ocean_vertical_diffusivity',0.0 )#)1.2e-5) #vertical diffusion coeff
# o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""

o.list_config()

###############################
# MODEL
###############################
# t0 = datetime(2020,1,1)
t0=reader_wind_wrf.start_time
# t0=reader_wind_era.start_time

# XY = [-4.65092,48.31261]
XY = [-4.562339,48.343159] 

o.seed_elements(lon = XY[0],#-4.8, 
                lat = XY[1],#48.3, 
                number = 100,
                z=-0.01 , # should be slighty<0 
                terminal_velocity= 0.0, # in m/s 
                time=t0,#., # one-off release
                wind_drift_factor=0.03 , #
                current_drift_factor=1 , 
                ) #

# check interpolation from readers
if False:
    depth = o.env.get_environment(['sea_floor_depth_below_sea_level'],
                                    lon=np.array([-4.8]),
                                    lat=np.array([-48.3]),
                                    z=np.array([0.0]),
                                    time= t0,
                                    profiles=None)
    depth = np.array(depth[0].tolist()).squeeze()#.reshape(xx.shape)

    u10 = o.env.get_environment(['x_wind'],
                                    lon=np.array([-4.8]),
                                    lat=np.array([-48.3]),
                                    z=np.array([0.0]),
                                    time= t0,
                                    profiles=None)


# Running model
o.run(time_step=900,
      stop_on_error=True,
	  end_time = t0 + timedelta(hours=12.0),  #21.0, # schism_native_sub.start_time + timedelta(days=2.0),) #
      time_step_output=timedelta(minutes=.5*60), # daily 
	  outfile='OUTPUT_opendrift_proto_wrf.nc'
)
    #   export_variables=['lon', 'lat','status','age_seconds'])

# Print and plot results
print(o)
import pdb;pdb.set_trace()

o.animation(fast=True, color='x_wind',show_trajectories=True,markersize=40)


o.animation(fast=True,
            filename = './outputs_opendrift/proto_surface_1km.gif',
            shapefiles=['../shp_files/release_test.shp','../shp_files/target_survey.shp']) # wont work if matplotlib was already imported


import opendrift 
o = opendrift.open('./outputs_opendrift/gigablue_surface.nc') # 3% windage
o1 = opendrift.open('./outputs_opendrift/gigablue_surface_nowindage.nc')
o.animation(fast=True,compare=o1,legend = []) # will fail if matplotlib was already imported


o = opendrift.open('./outputs_opendrift/gigablue_surface_proto.nc')
o.plot(fast=True)
o.animation(fast=True,markersize=1,shapefiles=['../shp_files/release_test.shp','../shp_files/target_survey.shp']) # wont work if matplotlib was already imported

o.animation(fast=True, color='z',show_trajectories=True,markersize=40)
o.animation_profile(markersize=40) #color='z'

# export to file
o.plot(fast=True,show_trajectories=True,filename = 'frame_release_settling.png')
o.animation(fast=True, color='z',show_trajectories=True,markersize=40,filename = 'frame_release_settling.gif')
o.animation_profile(markersize=40,filename = 'frame_release_settling_profile.gif') #color='z'



