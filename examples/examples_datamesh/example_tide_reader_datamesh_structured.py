import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_datamesh_regular_cons

###############################
# MODEL
###############################
o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
###############################
# READERS
###############################

reader_landmask = reader_global_landmask.Reader() 

datamesh_regular_cons = reader_datamesh_regular_cons.Reader(
      filename = '/home/simon/calypso_science/tide_grids/oceanum_2km.zarr',)  # native coordinate system is lon/lat

o.add_reader([reader_landmask,datamesh_regular_cons]) #
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding
o.set_config('drift:horizontal_diffusivity', 0.0) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)

time_run = [datetime(2024,1,1), datetime(2024,1,1) + timedelta(hours=24)]

# Seed elements at defined positions, depth and time
# 
# in Cook Strait
o.seed_elements(lon=174.572107,
                lat=-41.499003, 
                radius=500, # 1km radius 
                number=1000,
                z=np.linspace(0,-10, 1000), 
                time=time_run) # this will be a continuous release over that time vector

# Running model
o.run(stop_on_error = True,
      time_step=900, 
        end_time = time_run[-1],
        time_step_output = 1800.,
        outfile='test_datamesh_regular_cons.nc')

import pdb;pdb.set_trace()

o.plot(fast=True,filename='test_datamesh_regular_cons.png')
o.animation(fast=True,filename='test_datamesh_regular_cons.gif')
