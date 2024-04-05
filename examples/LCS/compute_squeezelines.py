import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift

# >> next step is to compute squeezelines, uisng saved variables C11,C12,C22
# see https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L12

# Try to compute squeezelines from the ds_lcs_new
import sys
sys.path.append('/home/simon/code/github/opendrift_simon/examples/LCS')
from cLCS_tools import compute_cLCS_squeezelines
import xarray as xr

ds_lcs_new = xr.open_dataset('ds_lcs_new.nc')

obj=compute_cLCS_squeezelines(ds_lcs_new)
obj.run()