#!/usr/bin/env python
"""
Leeway crosswind direction
==========================

Idealised test with constant 10 m/s wind blowing towards north, and no current.
Leeway coefficients are given for drift to the right (CWR) and left (CWL)
of downwind, with positive crosswind leeway to the right of downwind.

LIFE-RAFT-SB-10 (1-man raft with canopy) is asymmetric:

- right: slope 0.5 %, offset 7.0 cm/s -> about 12 cm/s towards east
- left: slope 0.1 %, offset -6.2 cm/s -> about 5.2 cm/s towards west

After 24 hours without jibing, right-drifting elements should thus be about 10.4 km
east and left-drifting elements about 4.5 km west of the downwind axis.
"""

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from opendrift.models.leeway import Leeway

object_type = 24  # LIFE-RAFT-SB-10
o = Leeway(loglevel=50)
o.set_config('environment:constant', {'x_sea_water_velocity': 0, 'y_sea_water_velocity': 0,
                'x_wind': 0, 'y_wind': 10, 'land_binary_mask': 0})
o.seed_elements(lon=4, lat=60, time=datetime(2020, 1, 1), number=2000,
                object_type=object_type, jibe_probability=0)
o.run(duration=timedelta(hours=24), time_step=900)

#%%
# Crosswind and downwind distance from the seed position, in km
x = (o.elements.lon - 4) * 111.2 * np.cos(np.radians(60))
y = (o.elements.lat - 60) * 111.2
ori = o.elements.orientation
p = o.leewayprop[object_type]
to_km = .01 * 24 * 3600 / 1000  # cm/s over 24 hours
expected = {0: (p['CWRSLOPE'] * 10 + p['CWROFFSET']) * to_km,
            1: (p['CWLSLOPE'] * 10 + p['CWLOFFSET']) * to_km}

fig, ax = plt.subplots(figsize=(7, 7))
for orientation, name, color in ((0, 'right', 'tab:red'), (1, 'left', 'tab:blue')):
    ind = ori == orientation
    print(f'{name:5s}: mean crosswind {x[ind].mean():5.1f} km '
          f'(expected {expected[orientation]:5.1f} km)')
    ax.scatter(x[ind], y[ind], s=2, c=color, label=f'{name} of downwind')
    ax.axvline(expected[orientation], color=color, ls='--')
    ax.plot(x[ind].mean(), y[ind].mean(), 'X', c=color, mec='k', ms=12)
ax.plot(0, 0, 'k*', markersize=12)
ax.annotate('', xy=(0, 8), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2))
ax.axvline(0, color='gray', lw=.5)
ax.set_xlabel('Crosswind (east) [km]')
ax.set_ylabel('Downwind (north) [km]')
ax.set_title('LIFE-RAFT-SB-10, 10 m/s wind towards north, 24 h\n'
             'X = modelled mean, dashed = expected from OBJECTPROP.DAT')
ax.set_aspect('equal')
ax.legend(loc='lower left', markerscale=5)
plt.show()
