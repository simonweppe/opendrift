#!/usr/bin/env python
"""
Element dependent environment
=============================
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from opendrift.models.oceandrift import OceanDrift
import trajan as ta

#%%
# OpenDrift elements have properties such as lon, lat, z, size etc.
# These elements are moved and changed based on environment properties such as current, wind, temperature etc.
# In principle the element properties do not affect the environment, however,
# for sensitivity studies it can be of interest to let different elements of 
# the same simulation be exposed to different environment.
# This is made possible by specifying the (constant) environment values
# for all elements of a seed call as illustrated below


#%%
# First an example with two different seedings with two different environments
o = OceanDrift(loglevel=20)
# First seeding 200 elements that will be exposed to eastward current and a horizontal diffusivity of 10 m2/s
number = 200
o.seed_elements(lon=-60, lat=40, time=datetime(2022,1,1), number=number, radius=10,
                environment={'horizontal_diffusivity': 10,
                             'x_sea_water_velocity': 1,
                             'y_sea_water_velocity': 0})
# Then seeding 100 elements that will be exposed to northward current and less diffusivity (1 m2/s)
number = 100
o.seed_elements(lon=-60, lat=40, time=datetime(2022,1,1), number=number, radius=10,
                environment={'horizontal_diffusivity': 1,
                             'x_sea_water_velocity': 0,
                             'y_sea_water_velocity': .5})
o.run(steps=10)
o.plot()

#%%
# Second example with a single seeding where each element will be exposed to different diffusivity values
o = OceanDrift(loglevel=20)
# Seeding 1000 elements that will be exposed to north-eastward current with diffusivities ranging from 0 to 50 m2/s
number = 1000
diffusivity_values = [0, 1, 5, 10, 50]
# Repeating values so that 200 elements get each diffusivity
diffusivities = np.repeat(diffusivity_values, number/len(diffusivity_values))

o.seed_elements(lon=-60, lat=40, time=datetime(2022,1,1), number=number, radius=10,
                environment={'horizontal_diffusivity': diffusivities,
                             'x_sea_water_velocity': .2,
                             'y_sea_water_velocity': .2})
ds = o.run(steps=10)
ds.traj.plot(land=None, margin=0)
# Plotting the convex hull around end positions separately for each diffusivity value
ds = ds.isel(time=-1)
colors = plt.cm.jet(np.linspace(0, 1, 5))
for d, color in zip(diffusivity_values, colors):
    ds.where(ds.horizontal_diffusivity==d).traj.plot.convex_hull(label=f'Diffusivity {d} m2/s', color=color)
plt.legend()
plt.show()

#%%
# The above could also be achieved by performing separate simulations for each value of diffusivity,
# but with more computational overhead/time and more complexity
