
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from opendrift.readers import reader_schism_native
# from opendrift.models.corallarvae import CoralLarvae
from opendrift.models.oceandrift import OceanDrift
import shapefile
import utm
from opendrift.readers import reader_shape
import cartopy.io.shapereader as shpreader
import argparse
import re


schism_path = 'schout_77*.nc'
landmask_path = 'schism_landmask.shp'
# SCHIMS data with 3D field of vertical diffusivities
reader0 = reader_schism_native.Reader(schism_path,proj4='+proj=utm +zone=4 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',use_true_outline=False)
reader_landmask = reader_shape.Reader.from_shpfiles(landmask_path)
o = OceanDrift(loglevel=0)
# if we want to just do a 3D interp of vertical diffusion fields, but not full profile interpolation (gradient of profile is used in vertical_mixing otheriwse)
# o.required_variables['ocean_vertical_diffusivity']['profiles'] = False

o.add_reader([reader_landmask,reader0])

# specific config on vertical mixing 
o.set_config('drift:vertical_mixing', True) 
o.set_config('vertical_mixing:diffusivitymodel','environment')
o.set_config('drift:profiles_depth',50)  # depth until which we use model-interpolated diffusion
o.set_config('environment:fallback:ocean_vertical_diffusivity', 1.2e-5) # used below <profiles_depth>  
o.set_config('drift:horizontal_diffusivity',0.0) 
#
o.set_config('general:use_auto_landmask', False) 
o.set_config('environment:fallback:x_wind', 0.0)
o.set_config('environment:fallback:y_wind', 0.0)
o.set_config('environment:fallback:x_sea_water_velocity', 0.0)
o.set_config('environment:fallback:y_sea_water_velocity', 0.0)
o.set_config('environment:fallback:sea_floor_depth_below_sea_level', 10000.0)
o.set_config('seed:ocean_only', False) 
o.set_config('drift:advection_scheme','runge-kutta4')  
o.set_config('general:seafloor_action', 'lift_to_seafloor')
o.set_config('general:coastline_action','previous')
o.set_config('drift:max_age_seconds', 3600*24*28)

o.list_config() 



##############################
#SEED PARTICLES
##############################
o.seed_elements(-157.79787475019606,20.374122045083183,number = 50, radius = 20, z = -1, time = reader0.start_time)

o.run(stop_on_error=False,
    end_time=reader0.start_time + timedelta(hours=11.),  #reader0.end_time,
    time_step=900, 
    time_step_output = 1800.0,
    # export_variables = ['trajectory', 'time', 'age_seconds', 'lon', 'lat', 'z'],
    outfile= f'variable_diffusion.nc')

import pdb;pdb.set_trace()
import opendrift;o=opendrift.open('variable_diffusion.nc');
o.animation(color='ocean_vertical_diffusivity')
o.animation_profile(color='ocean_vertical_diffusivity')
o.plot_property('ocean_vertical_diffusivity',) #filename='vertical_diffusivity_depth.png')
o.plot(linecolor='ocean_vertical_diffusivity',)
import pdb;pdb.set_trace()
