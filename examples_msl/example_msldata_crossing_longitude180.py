#!usrbinenv python

# This scripts tests what happens when particle cross the 180degW line and that :
# > the READER has longitude between [0-360]
# 

import numpy as np
from datetime import datetime, timedelta
from opendrift.readers import reader_basemap_landmask
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_netCDF_MetOcean
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.interpolation import ReaderBlock

###############################
# MODEL SELECTION
###############################

o = OceanDrift(loglevel=0)  # Set loglevel to 0 for debug information
###############################
# READERS
###############################
reader_cfsr = reader_netCDF_MetOcean.Reader('E:/metocean/0440_Tawhaki1_drillcuttings/opendrift_modelling/cfsr_ocean_nz_20040101.nc') # data goes across the 180deg longitude, and use 0<lon<360

# Making customised landmask (Basemap)

reader_basemap = reader_basemap_landmask.Reader(
                llcrnrlon=160.25, llcrnrlat=-50.75, 
                urcrnrlon=189.75, urcrnrlat=-33.02,   
                resolution='l', projection='merc') # resolution can be c (crude, the default), l (low), i (intermediate), h (high), f (full)    

o.max_speed =10.0 # must be before the add_readers() function !
o.add_reader([reader_basemap,reader_cfsr])

o.fallback_values['x_wind'] = 0. # strong westerly wind to make particle drift past 180degT and check behaviour
o.fallback_values['y_wind'] = 0.
# o.fallback_values['x_wind'] = 0 # strong westerly wind to make particle drift past 180degT and check behaviour
# o.fallback_values['y_wind'] = 0

###############################
# PARTICLE SEEDING
###############################

# Seeding some particles
# lons = np.linspace(3.5, 5.0, 100)
# lats = np.linspace(60, 61, 100)
# 
#  Point release
lon = 181.0; lat = -45.5;z=-16.0
#  Rectangle release
lons = np.linspace(179.0,181.0, 100) 
lats = np.linspace(-45.,-46., 10)
# these lon/lat spread across both roms_nz and cfsr data
depth = -np.linspace(0.,0., 1000) # surface=0, negative down

lons, lats = np.meshgrid(lons, lats)

lons = lons.ravel()
lats = lats.ravel()

# CHECK READER INTERPOLATION 
#  seems to return correct velocities provided that input lat/lon are using same covention as reader itself i.e. 0<lon<360, or -180<lon<180
# cfsr
# v = reader_cfsr.get_variables(['y_sea_water_velocity', 'x_sea_water_velocity'], time=reader_cfsr.start_time, x=lons, y=lats, z=depth, block=True)
# b = ReaderBlock(v, interpolation_horizontal='linearND')
# env, prof = b.interpolate(lons,lats,depth, ['y_sea_water_velocity', 'x_sea_water_velocity'])
# print env
# interpolation seems smooth across the 180 degT line
# import matplotlib.pyplot as plt
# plt.ion()
# plt.plot(lons,env1['x_sea_water_velocity'],'.')
# plt.show()
# plt.plot(lons,env1['y_sea_water_velocity'],'+')
# plt.show()
#
# Seed oil elements at defined position and time

o.seed_elements(lons, lats, z=depth, radius=0, number=1000, time = reader_cfsr.start_time, wind_drift_factor = 0.5) # large wind_drift_factor to make particles drift past 180 degT

import pdb;pdb.set_trace()

if False:
    # tests
    o.elements_scheduled # 50 elements have lon<180, 50 elements have lon>180
    # check interpolatio of first time step
    v0 = reader_cfsr.get_variables(['y_sea_water_velocity', 'x_sea_water_velocity'], time=reader_cfsr.start_time, x=o.elements_scheduled.lon, y=o.elements_scheduled.lat, z=depth, block=True)
    # o.elements_scheduled.lon>180 are taken care of internally in get_variables()
    b0 = ReaderBlock(v0, interpolation_horizontal='linearND')
    import pdb;pdb.set_trace()
    # try to convert coordinates of reader block ?
    # b0.x[np.where(b0.x>180)] = b0.x[np.where(b0.x>180)]-360
    # env0, prof0 = b0.interpolate(o.elements_scheduled.lon,o.elements_scheduled.lat,depth, ['y_sea_water_velocity', 'x_sea_water_velocity'])
    # or try convert the queried x,y of particles instead 
    o.elements_scheduled.lon[np.where(o.elements_scheduled.lon<0)] = 360 + o.elements_scheduled.lon[np.where(o.elements_scheduled.lon<0)]
    env1, prof1 = b0.interpolate(o.elements_scheduled.lon,o.elements_scheduled.lat,depth, ['y_sea_water_velocity', 'x_sea_water_velocity'])
    import matplotlib.pyplot as plt
    plt.ion()
    plt.plot(lons,env0['x_sea_water_velocity'],'.')

# PHYSICS
###############################
# o.set_config('driftcurrent_uncertainty', .1)
o.set_config('drift:scheme','runge-kutta4')
###############################
# RUN 
###############################
# Running model (until end of driver data)
o.run(time_step=1800, end_time = reader_cfsr.start_time + timedelta(days = 3), outfile='opendrift_test_longitude180_crossing.nc',time_step_output = 1800)  
# the start time is define by seed_elements, the end_time is defined by either steps=number of step, duration = timedelta, or end_time= datetime

###############################
# PLOTS  ANIMATION
###############################
# Print and plot results
print o
o.animation()
o.plot()
