# to test the 3D runs using SCHISM dataset interpolated to Z-levels
import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_datamesh_zlevels
from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
import xarray as xr

##################################################################################################################
# MODEL
##################################################################################################################
o_base = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
##################################################################################################################
# READERS
##################################################################################################################
reader_landmask = reader_global_landmask.Reader()

# NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

reader_schism_datamesh_zlevels = reader_schism_datamesh_zlevels.Reader(
    filename = './oceanum_ocean_nz_schism_zlevels_v1/oceanum_ocean_nz_schism_zlevels_v1_native_2020010*.nc',
    proj4 = proj4str_nztm, # projection that will be used to convert from lon/lat to cartesian (needed for KDtree to work)
    )

o_base.add_reader([reader_landmask,reader_schism_datamesh_zlevels]) #
o_base.set_config('general:use_auto_landmask', False) # we use the default one here which is fine for large scale
o_base.set_config('general:coastline_action', 'previous') # ''  prevent particles stranding 
o_base.set_config('drift:horizontal_diffusivity', 0.0) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)
o_base.set_config('drift:advection_scheme', 'runge-kutta4') # Note that Runge-Kutta here makes a difference to Euler scheme
o_base.set_config('seed:ocean_only', False) # # we dont want to reseed particles to nearest location
o_base.disable_vertical_motion()  #Deactivate any vertical processes/advection"""
# ##################################################################################################################
# SEEDING
# ##################################################################################################################
# Simple seeding to test run
# Release at different depth in water column
# 
# Note: in the interpolated zlevel files, the -0.5m level includes data from -0.5 below surface 
# (NOT -0.5 relative to MSL), which means it is continuous and representative to surface currents at all times 
# ##################################################################################################################
X=174.542839
Y=-40.808506
for ii,z_rel in enumerate([-.5, -10.,-20., -40.,-60.,-100.,-150.]):
    o_base.seed_elements(lon=X, lat=Y,  number=1,
                        z=z_rel, 
                        time=reader_schism_datamesh_zlevels.start_time,
                        origin_marker=ii) 

# Running model
o_base.run(time_step=900,
     end_time = reader_schism_datamesh_zlevels.start_time+timedelta(days=2))
     # outfile='schism_native_output.nc')
import pdb;pdb.set_trace()
o_base.plot(fast=True,color='origin_marker')
o_base.animation(fast=True,color='z') # ,color='origin_marker')
