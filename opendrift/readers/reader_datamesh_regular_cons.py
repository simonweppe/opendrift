# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.


##########################################################################
# This reader supports structured tidal constituent grid from
# Oceanum's Datamesh. https://docs.oceanum.io/datamesh/index.html
# Gridded fields of elev, u, v are reconstructed at each timestep 
# using the python package oceantide (https://github.com/oceanum/oceantide/)
# 
# 
# The reader is starting from the reader_netCDF_generic.py as base.
# 
# We overload the _get_interpolated_variables_() function to provide 
# tidal data directly at the particle locations.
# 
# Author: Simon Weppe. Calypso Science New Zealand
##########################################################################

# Copyright 2015, Knut-Frode Dagestad, MET Norway,  
# Copyright 2024, Simon Weppe. Calypso Science New Zealand

from datetime import datetime
import pyproj
import numpy as np
from netCDF4 import num2date
import logging
logger = logging.getLogger(__name__)

from opendrift.readers.basereader import BaseReader, StructuredReader
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers.interpolation.structured import ReaderBlock
import xarray as xr
import oceantide


standard_name_mapping_datamesh = {
    'u': 'x_sea_water_velocity', 
    'v': 'y_sea_water_velocity',
    'dep': 'sea_floor_depth_below_sea_level',
    'h' : 'sea_surface_height'}

# do an inverted version
standard_name_mapping_datamesh_invert = {v: k for k, v in standard_name_mapping_datamesh.items()}


# class Reader(StructuredReader, BaseReader):
class Reader(reader_netCDF_CF_generic.Reader):
    """
    A reader for `CF-compliant <https://cfconventions.org/>`_ netCDF files. It can take a single file, a file pattern, a URL or an xarray Dataset.

    Args:
        :param filename: A single netCDF file, a pattern of files, or a xr.Dataset. The
                         netCDF file can also be an URL to an OPeNDAP server.
        :type filename: string, xr.Dataset (required).

        :param name: Name of reader
        :type name: string, optional

        :param proj4: PROJ.4 string describing projection of data.
        :type proj4: string, optional

        kwargs      : use_log_profile : use log profile to extrpolate current at any level in water column.
                 z0 : roughness height in meters (default 0.0001m for sandy areas)

    Example:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader("arome_subset_16Nov2015.nc")

    Several files can be specified by using a pattern:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader("*.nc")

    An OPeNDAP URL can be used:

    .. code::

       from opendrift.readers.reader_netCDF_CF_generic import Reader
       r = Reader('https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc')

    A xr.Dataset or a zarr dataset in an object store with auth can be used:

    .. code::

        from opendrift.readers.reader_netCDF_CF_generic import Reader
        r = Reader(ds, zarr_storage_options)
    """

    def __init__(self, filename=None, zarr_storage_options=None, name=None, proj4=None,
                 standard_name_mapping=standard_name_mapping_datamesh, ensemble_member=None,**kwargs):
        
        # variable name mapping in datamesh constituent grid


        # Run constructor of parent Reader class
        # specify correct variable name mapping
        super(Reader,self).__init__(filename=filename, 
                                    zarr_storage_options=None, 
                                    name=None, 
                                    proj4=None, 
                                    standard_name_mapping=standard_name_mapping_datamesh, # use the datamesh variable mapping
                                    ensemble_member=None,
                                    **kwargs)
                
        # add some reader-specific options
        if 'use_log_profile' in kwargs:
            self.use_log_profile = kwargs['use_log_profile']
            if self.use_log_profile :
                
                if 'z0' in kwargs:
                    self.z0 = kwargs['z0']
                else:
                    self.z0 = 0.0001 # default
                logger.debug('Using log profile for current extrapolation in water column, with roughness height %s' % self.z0)
        else:
            self.use_log_profile = False

        # use dummy start/end times instead, to make it always valid time-wise (needed for some check in get_environment() )
        self.start_time = datetime(1000,1,1) 
        self.end_time = datetime(3000,1,1) 

        # by default we activate the derivation of land_binary_mask from 'sea_floor_depth_below_sea_level
        # https://github.com/OpenDrift/opendrift/blob/master/opendrift/readers/basereader/variables.py#L443
        self.activate_environment_mapping('land_binary_mask_from_ocean_depth')

    def nearest_time(self, time):
        """ overloads version from variables.py
        
        Original function : Return nearest times before and after the requested time.

        Here : we return the input time as nearest time as tide can be generated for any time
        Note this will not lead to interpolation in _get_variables_interpolated_()
        as the "right on time" case will be identified

        Returns:
            nearest_time: datetime
            time_before: datetime
            time_after: datetime
            indx_nearest: int
            indx_before: int
            indx_after: int
        """
        nearest_time = time
        time_before = time
        time_after =time
        indx_nearest, indx_before, indx_after = None,None,None # these are not used in get_variables()
        return nearest_time, time_before, time_after,\
            indx_nearest, indx_before, indx_after
    
    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None,
                      indrealization=None):
        # this step is not needed anymore since we generate tidal data directly 
        # at particle positions in _get_variables_interpolated_()

        return None
    
    def _get_variables_interpolated_(self, variables, profiles, profiles_depth,
                                     time, reader_x, reader_y, z):
        print(time)
        # overloads the version from <structured.py>
        # 
        # Here we interpolate constituents to particle positions then generate tide signals 
        # (instead of interpolating from gridded fields created in get_variables() )

        # For global readers, we shift coordinates to match actual lon range

        if self.global_coverage():
            if self.lon_range() == '-180to180':
                logger.debug('Shifting coordinates to -180-180')
                reader_x = np.mod(reader_x + 180, 360) - 180
            elif self.lon_range() == '0to360':
                logger.debug('Shifting coordinates to 0-360')
                reader_x = np.mod(reader_x, 360)
        elif self.proj.crs.is_geographic and self.xmin>0:
            logger.debug('Modulating longitudes to 0-360 for self.name')
            reader_x = np.mod(reader_x, 360)

        # Find reader time_before/time_after
        time_nearest, time_before, time_after, i1, i2, i3 = \
            self.nearest_time(time)
        logger.debug('Reader time:\n\t\t%s (before)\n\t\t%s (after)' %
                     (time_before, time_after))

        # For variables which are not time dependent, we do not care about time
        static_variables = [
            'sea_floor_depth_below_sea_level', 'land_binary_mask'
        ]
        if time == time_before or all(v in static_variables
                                      for v in variables):
            time_after = None

        if profiles is not None:
            # If profiles are requested for any parameters, we
            # add two fake points at the end of array to make sure that the
            # requested block has the depth range required for profiles
            mx = np.append(reader_x, [reader_x[-1], reader_x[-1]])
            my = np.append(reader_y, [reader_y[-1], reader_y[-1]])
            mz = np.append(z, [profiles_depth[0], profiles_depth[1]])
        else:
            mx = reader_x
            my = reader_y
            mz = z

        # Interpolate constituents to particle positions, then generate tide data
        # 
        # Note : self.Dataset.interp(lon=reader_x, lat=reader_y).tide.predict(times=time) 
        # returns a matrix (reader_x_size,reader_y_size) which is not what we want  
        #  
        # Here, we need to use advanced indexing see below
        # https://stackoverflow.com/questions/55034347/extract-interpolated-values-from-a-2d-array-based-on-a-large-set-of-xy-points
        # http://xarray.pydata.org/en/stable/user-guide/interpolation.html#advanced-interpolation
        
        lon_id = xr.DataArray(reader_x, dims='z')
        lat_id = xr.DataArray(reader_y, dims='z') 
        
        if 'x_sea_water_velocity' in variables :
            tide_pred = self.Dataset.interp(lon=lon_id, lat=lat_id).tide.predict(times=time) 
            # the <env> variable to return is a dict such as
            # env =  {'sea_floor_depth_below_sea_level' : np.array(), ...}
            # 
            # package data to dictionary 
            env = {}
            for var in variables:
                env[var] = np.ma.masked_invalid(tide_pred[standard_name_mapping_datamesh_invert[var]])
        
        else: # static variables, like depth, used to estimate land_binary_mask via land_binary_mask_from_ocean_depth()
            if variables == ['sea_floor_depth_below_sea_level'] : # only depth can be requested as static variables
                depth = self.Dataset['dep'].interp(lon=lon_id, lat=lat_id)
                # package to dictionary
                env = {}
                env['sea_floor_depth_below_sea_level'] = np.ma.masked_invalid(depth)
            else:
                # should not happen for now
                import pdb;pdb.set_trace()


        # not supporting profiles for now - set to None
        env_profiles = None

        return env, env_profiles