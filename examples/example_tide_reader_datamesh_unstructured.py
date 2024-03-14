import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_schism_datamesh_cons

###############################
# MODEL
###############################
o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
###############################
# READERS
###############################

reader_landmask = reader_global_landmask.Reader() 

schism_datamesh_cons = reader_schism_datamesh_cons.Reader(
	filename = 'calypso-tidalcons-hauraki-v1',)  # native coordinate system is lon/lat
# this will include the mesh boundary polygons and interior islands

# >> see how we can use it as land_bindary_mask
import pdb;pdb.set_trace()
# schism_native.plot_mesh()
o.add_reader([reader_landmask,schism_datamesh_cons]) #
o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
o.set_config('general:coastline_action', 'previous') # prevent particles stranding
o.set_config('drift:horizontal_diffusivity', 0.1) # Switch on horizontal diffusivity. Set this at 0.1 m2/s (https://journals.ametsoc.org/view/journals/atot/22/9/jtech1794_1.xml)

# time_run = [datetime.utcnow(), datetime.utcnow() + timedelta(hours=12)]
time_run = [datetime(2024,1,1), datetime(2024,1,1) + timedelta(hours=12)]

# Seed elements at defined positions, depth and time
o.seed_elements(lon=175.0060864, 
                lat=-36.5267795, 
                radius=250, 
                number=100,
                z=np.linspace(0,-10, 100), 
                time=time_run) # this will be a continuous release over that time vector

o.seed_elements(lon=174.80710933775802, 
                lat=-36.83665631119203, 
                radius=250, 
                number=100,
                z=np.linspace(0,-10, 100), 
                time=time_run) # this will be a continuous release over that time vector


# Running model
o.run(stop_on_error = True,
      time_step=600, 
	  end_time = time_run[-1],
      time_step_output = 1800.)
	  # outfile='schism_native_output.nc')
import pdb;pdb.set_trace()
o.plot(fast=True,filename='test_schism_datamesh_cons.png')
o.animation(fast=True,filename='test_schism_datamesh_cons.gif')

# >> need to find a way to use the land_binary mask information from the cons grid