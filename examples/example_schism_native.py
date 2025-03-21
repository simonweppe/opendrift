#!/usr/bin/env python
"""
SCHISM native reader
==================================
"""


import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_global_landmask
from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift

o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information

# <<<<<<< HEAD
# reader_custom = reader_landmask_custom.Reader(polygon_file = '/home/simon/Documents/GitHub/opendrift/examples/schism_marl_edit.shore')
# # reader_custom.plot() # check reader was correctly loaded, close figure to continue

# # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
# proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
# schism_native = reader_schism_native.Reader(
# 	filename = 'https://thredds.met.no/thredds/dodsC/metusers/knutfd/thredds/netcdf_unstructured_samples/schism_marl20080101_00z_3D.nc',
# 	proj4 = proj4str_nztm,
# 	use_3d = True)
# # schism_native.plot_mesh(variable = ['sea_floor_depth_below_sea_level']) # check reader was correctly loaded

# o.add_reader([reader_custom,schism_native])
# o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
# o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""
# =======

try:
    # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
    proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    schism_native = reader_schism_native.Reader(
        filename = 'https://thredds.met.no/thredds/dodsC/metusers/knutfd/thredds/netcdf_unstructured_samples/schism_marl20080101_00z_3D.nc',
        proj4 = proj4str_nztm,
        use_3d = True)
except:
    print('Thredds server not available, cannot run example')
    schism_native = None
# >>>>>>> upstream/master


if schism_native is not None:
    o.add_reader([schism_native])

    # Seed elements at defined positions, depth and time
    o.seed_elements(lon=174.046669, lat=-40.928116, radius=20, number=100,
                    z=np.linspace(0,-10, 100), time=schism_native.start_time)

#%%
# Running model
o.run(time_step=900,
	  end_time = schism_native.start_time+timedelta(days=0.1))
	  # outfile='schism_native_output.nc')

o.seed_elements(lon= 173.8839, lat=-40.9160, radius=20, number=100,
                z=np.linspace(0,-10, 100), time=schism_native.start_time)

o.seed_elements(lon=174.2940, lat=-41.0888, radius=20, number=100,
                z=np.linspace(0,-10, 100), time=schism_native.start_time)

#%%
# Running model
o.run(time_step=900,
        end_time = schism_native.start_time+timedelta(days=0.1))
        # outfile='schism_native_output.nc')

# Print and plot results
print(o)
o.plot(fast=True)
o.animation()
o.animation_profile()
