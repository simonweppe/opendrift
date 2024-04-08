import numpy as np
from datetime import timedelta, datetime
from opendrift.readers import reader_schism_native
from opendrift.readers import reader_constant
from opendrift.readers import reader_global_landmask
# from opendrift.readers import reader_landmask_custom
from opendrift.models.oceandrift import OceanDrift

# Try to compute squeezelines from the ds_lcs_new
import sys
sys.path.append('/home/simon/code/github/opendrift_simon/examples/LCS')
from cLCS_tools import compute_cLCS_squeezelines
import xarray as xr

###############################
# MODEL
###############################
if False:
    o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information
    reader_landmask = reader_global_landmask.Reader()
    proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84

    schism_native = reader_schism_native.Reader(
        filename = '/home/simon/calypso_science/projects/NP_disposalgrounds/opendrift_modelling/oceanum_ocean_nz_schism_v1_2D.nc',
        proj4 = proj_wgs84,
        use_3d = False,
        use_model_landmask = False)

    cst_reader_wind = reader_constant.Reader( {'x_wind': 0, 'y_wind': 0,}) # add a constant reader just to check it's used in the LCS computations >> OK

    o.add_reader([reader_landmask,schism_native,cst_reader_wind])
    o.set_config('general:use_auto_landmask', False) # prevent opendrift from making a new dynamical landmask with global_landmask
    o.disable_vertical_motion()  #Deactivate any vertical processes/advection"""

    # start time for LCS computation
    time_lcs_start  = [schism_native.start_time,schism_native.start_time +timedelta(hours=12.)] # can a single value or list of values
    integration_time = timedelta(hours=12)  # integration time to compute the LCS (using position at t0 and t0+integration_time)

    # new green-cauchy tensors - still need to add squeezelines
    lcs_new,ds_lcs_new = o.calculate_green_cauchy_tensor(
        reader     = schism_native,
        time       = time_lcs_start[0], # the start time of LCS computation ..can be a single value or list of values
        time_step  = timedelta(minutes=15), # time step of individual opendrift simulations
        duration   = integration_time,    
        delta      = 10000, # spatial step (in meter or degrees depending of reader coords) at which the particles will be seeded within domain
        domain     = [172.0, 174.0, -40.0, -38.0], # user-defined frame within reader domain [xmin, xmax, ymin, ymax], if None use entire domain
        ALCS       = True,  # attractive LCS, run backwards in time
        RLCS       = False, # repulsive LCS, run forward in time
        cartesian_epsg = 2193)
    ds_lcs_new.to_netcdf('ds_lcs_new.nc')
    import pdb;pdb.set_trace()


# see script to compute squeezeline here:
# 
# https://github.com/MireyaMMO/cLCS/blob/main/cLCS/make_cLCS.py#L12


ds_lcs_new = xr.open_dataset('ds_lcs_new.nc')
obj=compute_cLCS_squeezelines(ds_lcs_new)
obj.run()
import pdb;pdb.set_trace()

# squeezelines are saved here obj.pxt, obj.pyt

# now need to see how to plot correctly 
# 
# see code employed here : https://github.com/MireyaMMO/cLCS/blob/main/examples/01_cLCS_ROMS.ipynb