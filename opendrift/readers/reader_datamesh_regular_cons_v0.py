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
# along with OpenDrift.  If not, see <http://www.gnu.org/licenses/>.
#
#


##########################################################################
# This reader supports unstructured tidal constituent grid from
# Oceanum's Datamesh. https://docs.oceanum.io/datamesh/index.html
# Gridded fields of elev, u, v are reconstructed at each timestep 
# using the python package oceantide (https://github.com/oceanum/oceantide/)
# 
# 
# To test : interpolation of constituents to particle position rather
# than generating full grid then interpolating from that 
# 
# Author: Simon Weppe. Calypso Science New Zealand
##########################################################################

import logging
logger = logging.getLogger(__name__)

import numpy as np
from datetime import datetime
from future.utils import iteritems
from netCDF4 import Dataset, MFDataset, num2date
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree #cython-based KDtree for quick nearest-neighbor search
# consider using https://github.com/storpipfugl/pykdtree for KDtree computations - test efficiency
import pyproj
from opendrift.readers.basereader import BaseReader, StructuredReader
from opendrift.readers.basereader.consts import *
import xarray as xr
import shapely
import oceantide

class Reader(BaseReader,StructuredReader):

    def __init__(self, filename=None, name=None, **kwargs):
        """Initialise reader_netCDF_CF_unstructured_SCHISM

        Args:
            filename    :   name of unstructured constituent grid from Oceanum's Datamesh

            name        :   name of reader - optional, taken as filename if not input
                            o.readers['name']
            
            kwargs      : use_log_profile : use log profile to extrpolate current at any level in water column.
                          z0 : roughness height in meters (default 0.0001m for sandy areas)

        """
        if filename is None:
            raise ValueError('Need filename as argument to constructor')
        filestr = str(filename)
        if name is None:
            self.name = filestr
        else:
            self.name = name

        # Default interpolation method, see function interpolate_block()
        self.interpolation = 'linearNDFast'
        self.convolve = None  # Convolution kernel or kernel size
        
        # [name_used_in_schism : equivalent_CF_name]
        schism_mapping = {
            'u': 'x_sea_water_velocity', 
            'v': 'y_sea_water_velocity',
            'dep': 'sea_floor_depth_below_sea_level',
            'h' : 'sea_surface_height'}

        self.return_block = True

        try:
            # Open file, check that everything is ok
            logger.info('Opening dataset: ' + filestr)
            if ('.nc' not in filestr) :
                logger.info('Opening files with open_zarr')
                self.Dataset = xr.open_zarr(filestr)

                # make sure both lon and lat have their attributes
                if len(self.Dataset.lon.attrs) == 0:
                    self.Dataset.lon.attrs = {'long_name': 'Longitude', 'standard_name': 'longitude', 'units': 'degree_east'}
                if len(self.Dataset.lat.attrs) == 0:
                    self.Dataset.lat.attrs = {'long_name': 'Latitude', 'standard_name': 'latitude', 'units': 'degree_north'}                   
                
            else:
                logger.info('Opening file with open_dataset')
                import pdb;pdb.set_trace() # not tested yet
                self.Dataset = xr.open_dataset(filestr)

            # # need to edit the cons name for correct use in oceantide later on
            # self.Dataset['con']=[x.strip().upper() for x in self.Dataset['cons'].values]

        except Exception as e:
            raise ValueError(e)

        # Define projection of input data - will always be lon/lat
        self.proj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' #'+proj=latlong'
        
        # use dummy start/end times instead, to make it always valid time-wise
        self.start_time = datetime(1000,1,1) 
        self.end_time = datetime(3000,1,1) 
        
        logger.debug('Finding coordinate variables.')
        # Find x, y and z coordinates
        for var_name in self.Dataset.variables:

            if var_name in ['con']:
                continue

            var = self.Dataset.variables[var_name]

            attributes = var.attrs
            att_dict = var.attrs
            # attributes = var.ncattrs()
            standard_name = ''
            long_name = ''
            axis = ''
            units = ''
            CoordinateAxisType = ''
            # add checks on projection here ? 
            # as in reader_netCDF_CF_generic.py
            if 'standard_name' in attributes:
                standard_name = att_dict['standard_name']
            if 'long_name' in attributes:
                long_name = att_dict['long_name']
            if 'axis' in attributes:
                axis = att_dict['axis']
            if 'units' in attributes:
                units = att_dict['units']
            if '_CoordinateAxisType' in attributes:
                CoordinateAxisType = att_dict['_CoordinateAxisType']
            
            
            if standard_name == 'longitude' or \
                    long_name == 'longitude' or \
                    var_name == 'longitude' or \
                    standard_name == 'Longitude' or \
                    long_name == 'Longitude' or \
                    var_name == 'Longitude' or \
                    axis == 'X' or \
                    CoordinateAxisType == 'Lon' or \
                    standard_name == 'projection_x_coordinate':
                self.xname = var_name
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    unitfactor = 1000
                else:
                    unitfactor = 1
                var_data = var.values
                x = var_data*unitfactor
                self.unitfactor = unitfactor
                self.numx = var.shape[0]
            if standard_name == 'latitude' or \
                    long_name == 'latitude' or \
                    var_name == 'latitude' or \
               standard_name == 'Latitude' or \
                    long_name == 'Latitude' or \
                    var_name == 'Latitude' or \
                    axis == 'Y' or \
                    CoordinateAxisType == 'Lat' or \
                    standard_name == 'projection_y_coordinate':
                self.yname = var_name
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    unitfactor = 1000
                else:
                    unitfactor = 1
                var_data = var.values
                y = var_data*unitfactor
                self.numy = var.shape[0]
            if standard_name == 'sea_floor_depth_below_mean_sea_level' or axis == 'Z':
                var_data = var.values
                if 'positive' not in var.attrs or \
                        var.attrs['positive'] == 'up':
                    self.z = var_data
                else:
                    self.z = -var_data

            # there will be no time here
            if standard_name == 'time' or axis == 'T' or var_name == 'time':
                var_data = var.values
                # Read and store time coverage (of this particular file)
                time = var_data
                time_units = units
                # self.times = num2date(time, time_units)
                # convert from numpy.datetime64 to datetime
                self.times = [datetime.utcfromtimestamp((OT -
                    np.datetime64('1970-01-01T00:00:00Z')
                        ) / np.timedelta64(1, 's')) for OT in time]

                self.start_time = self.times[0]
                self.end_time = self.times[-1]
                if len(self.times) > 1:
                    self.time_step = self.times[1] - self.times[0]
                else:
                    self.time_step = None
         
            if standard_name == 'tidal_constituent' : #or axis == 'T' or var_name == 'time':
                # load the tidal consituent data 
                # Not needed > the tide prediction will be handled directly by oceantide using xarray object
                pass
                
        if 'x' not in locals():
            raise ValueError('Did not find x-coordinate variable')
        if 'y' not in locals():
            raise ValueError('Did not find y-coordinate variable')

        self.x = x
        self.y = y
        # add the delta_x/y to correctly identify indices in get_variables()
        self.delta_x = np.diff(self.x)[0]
        self.delta_y = np.diff(self.y)[0]
        
        # Run constructor of parent Reader class
        super(Reader, self).__init__()

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

        # by default we activate the derivation of land_binary_mask from 'sea_floor_depth_below_sea_level
        # https://github.com/OpenDrift/opendrift/blob/master/opendrift/readers/basereader/variables.py#L443
        self.activate_environment_mapping('land_binary_mask_from_ocean_depth')

        # Find all variables having standard_name
        self.variable_mapping = {}
        for var_name in self.Dataset.variables:
            if var_name in [self.xname, self.yname]: #'depth'
                continue  # Skip coordinate variables
            var = self.Dataset.variables[var_name]
            attributes = var.attrs
            att_dict = var.attrs

            if var_name in schism_mapping:                           
                self.variable_mapping[schism_mapping[var_name]] = str(var_name) 
                   
        self.variables = list(self.variable_mapping.keys())

        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()

        # Run constructor of parent Reader class
        super(Reader, self).__init__()
        
        # Dictionaries to store blocks of data for reuse (buffering)
        self.var_block_before = {}  # Data for last timestep before present
        self.var_block_after = {}   # Data for first timestep after present

    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None, block=False):
        # copied from reader_netCDF_generic
        # 
        # Option 1 : generate gridded tidal elevation/currents to be passed to 
        #            generic interpolation process (i.e. values at particle positions)
        #            interpolated from gridded fields
        # 
        # Option 2 : generate tidal elevation/currenst directly at particle poistions
        #            by interpolating constituents at particle positions

        requested_variables, time, x, y, z, _outside = self.check_arguments(
            requested_variables, time, x, y, z)
        
        # use nearest_time() from this reader (see below)
        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time) 
        
        # if hasattr(self, 'z') and (z is not None):
        #     # Find z-index range
        #     # NB: may need to flip if self.z is ascending
        #     indices = np.searchsorted(-self.z, [-z.min(), -z.max()])
        #     indz = np.arange(np.maximum(0, indices.min() - 1 -
        #                                 self.verticalbuffer),
        #                      np.minimum(len(self.z), indices.max() + 1 +
        #                                 self.verticalbuffer))
        #     if len(indz) == 1:
        #         indz = indz[0]  # Extract integer to read only one layer
        # else:
        indz = 0

        # if indrealization == None:
        #     if self.realizations is not None:
        #         indrealization = range(len(self.realizations))
        #     else:
        #         indrealization = None

        # Find indices corresponding to requested x and y
        if hasattr(self, 'clipped'):
            clipped = self.clipped
        else: clipped = 0
        
        if self.global_coverage():
            if self.lon_range() == '0to360':
                x = np.mod(x, 360)  # Shift x/lons to 0-360
            elif self.lon_range() == '-180to180':
                x = np.mod(x + 180, 360) - 180 # Shift x/lons to -180-180
        indx = np.floor(np.abs(x-self.x[0])/self.delta_x-clipped).astype(int) + clipped
        indy = np.floor(np.abs(y-self.y[0])/self.delta_y-clipped).astype(int) + clipped
        buffer = self.buffer  # Adding buffer, to cover also future positions of elements
        indy = np.arange(np.max([0, indy.min()-buffer]),
                         np.min([indy.max()+buffer, self.numy]))
        indx = np.arange(indx.min()-buffer, indx.max()+buffer+1)

        if self.global_coverage() and indx.min() < 0 and indx.max() > 0 and indx.max() < self.numx:
            logger.debug('Requested data block is not continuous in file'+
                          ', must read two blocks and concatenate.')
            indx_left = indx[indx<0] + self.numx  # Shift to positive indices
            indx_right = indx[indx>=0]
            if indx_right.max() >= indx_left.min():  # Avoid overlap
                indx_right = np.arange(indx_right.min(), indx_left.min())
            continuous = False
        else:
            continuous = True
            indx = np.arange(np.max([0, indx.min()]),
                             np.min([indx.max(), self.numx]))

        variables = {}

        # if tidal velocities are requested, we generate the flow field for that time
        if 'x_sea_water_velocity' in requested_variables:
            tide_pred = self.Dataset.tide.predict(times=time)

        for par in requested_variables:
            
            if hasattr(self, 'rotate_mapping') and par in self.rotate_mapping:
                logger.debug('Using %s to retrieve %s' %
                    (self.rotate_mapping[par], par))
                if par not in self.variable_mapping:
                    self.variable_mapping[par] = \
                        self.variable_mapping[
                            self.rotate_mapping[par]]
                    
            var = self.Dataset.variables[self.variable_mapping[par]]
            import pdb;pdb.set_trace()

            ensemble_dim = None
            if continuous is True:
                if True:  # new dynamic way
                    dimindices = {'x': indx, 'y': indy, 'time': indxTime, 'z': indz}
                    subset = {vdim:dimindices[dim] for dim,vdim in self.dimensions.items() if vdim in var.dims}
                    variables[par] = var.isel(subset)
                    # Remove any unknown dimensions
                    for dim in variables[par].dims:
                        if dim not in self.dimensions.values() and dim != self.ensemble_dimension:
                            logger.debug(f'Removing unknown dimension: {dim}')
                            variables[par] = variables[par].squeeze(dim=dim)
                    if self.ensemble_dimension is not None and self.ensemble_dimension in variables[par].dims:
                        ensemble_dim = 0  # hardcoded, may not work for MEPS
                else:  # old hardcoded way
                    if var.ndim == 2:
                        variables[par] = var[indy, indx]
                    elif var.ndim == 3:
                        variables[par] = var[indxTime, indy, indx]
                    elif var.ndim == 4:
                        variables[par] = var[indxTime, indz, indy, indx]
                    elif var.ndim == 5:  # Ensemble data
                        variables[par] = var[indxTime, indz, indrealization, indy, indx]
                        ensemble_dim = 0  # Hardcoded ensemble dimension for now
                    else:
                        raise Exception('Wrong dimension of variable: ' +
                                        self.variable_mapping[par])
                import pdb;pdb.set_trace()
            # The below should also be updated to dynamic subsetting
            else:  # We need to read left and right parts separately
                if var.ndim == 2:
                    left = var[indy, indx_left]
                    right = var[indy, indx_right]
                    variables[par] = np.ma.concatenate((left, right), 1)
                elif var.ndim == 3:
                    left = var[indxTime, indy, indx_left]
                    right = var[indxTime, indy, indx_right]
                    variables[par] = np.ma.concatenate((left, right), 1)
                elif var.ndim == 4:
                    left = var[indxTime, indz, indy, indx_left]
                    right = var[indxTime, indz, indy, indx_right]
                    variables[par] = np.ma.concatenate((left, right), 2)
                elif var.ndim == 5:  # Ensemble data
                    left = var[indxTime, indz, indrealization,
                               indy, indx_left]
                    right = var[indxTime, indz, indrealization,
                                indy, indx_right]
                    variables[par] = np.ma.concatenate((left, right), 3)

            variables[par] = np.asarray(variables[par])

            # Mask values outside domain
            variables[par] = np.ma.array(variables[par],
                                         ndmin=2, mask=False)
            # Mask extreme values which might have slipped through
            with np.errstate(invalid='ignore'):
                variables[par] = np.ma.masked_outside(
                    variables[par], -30000, 30000)

            # Ensemble blocks are split into lists
            if ensemble_dim is not None:
                num_ensembles = variables[par].shape[ensemble_dim]
                logger.debug(f'Num ensembles for {par}: {num_ensembles}')
                newvar = [0]*num_ensembles
                for ensemble_num in range(num_ensembles):
                    newvar[ensemble_num] = \
                        np.take(variables[par],
                                ensemble_num, ensemble_dim)
                variables[par] = newvar

        # Store coordinates of returned points
        try:
            variables['z'] = self.z[indz]
        except:
            variables['z'] = None
        if self.projected is True:
            variables['x'] = \
                self.Dataset.variables[self.xname][indx]*self.unitfactor
            variables['y'] = \
                self.Dataset.variables[self.yname][indy]*self.unitfactor
        else:
            variables['x'] = indx
            variables['y'] = indy
        variables['x'] = np.asarray(variables['x'], dtype=np.float32)
        variables['y'] = np.asarray(variables['y'], dtype=np.float32)

        variables['time'] = nearestTime

        # Rotate any east/north vectors if necessary
        if hasattr(self, 'rotate_mapping'):
            if self.y_is_north() is True:
                logger.debug('North is up, no rotation necessary')
            else:
                rx, ry = np.meshgrid(variables['x'], variables['y'])
                lon, lat = self.xy2lonlat(rx, ry)
                from opendrift.readers.basereader import vector_pairs_xy
                for vectorpair in vector_pairs_xy:
                    if vectorpair[0] in self.rotate_mapping and vectorpair[0] in variables.keys():
                        if self.proj.__class__.__name__ == 'fakeproj':
                            logger.warning('Rotation from fakeproj is not yet implemented, skipping.')
                            continue
                        logger.debug(f'Rotating vector from east/north to xy orientation: {vectorpair[0:2]}')
                        variables[vectorpair[0]], variables[vectorpair[1]] = self.rotate_vectors(
                            lon, lat, variables[vectorpair[0]], variables[vectorpair[1]],
                            pyproj.Proj('+proj=latlong'), self.proj)

        if hasattr(self, 'shift_x'):
            # "hidden feature": if reader.shift_x and reader.shift_y are defined,
            # the returned fields are shifted this many meters in the x- and y directions
            # E.g. reader.shift_x=10000 gives a shift 10 km eastwards (if x is east direction)
            if self.proj.crs.is_geographic:  # meters to degrees
                shift_y = (self.shift_y/111000)
                shift_x = (self.shift_x/111000)*np.cos(np.radians(variables['y']))
                logger.info('Shifting x between %s and %s' % (shift_x.min(), shift_x.max()))
                logger.info('Shifting y with %s m' % shift_y)
            else:
                shift_x = self.shift_x
                shift_y = self.shift_y
                logger.info('Shifting x with %s m' % shift_x)
                logger.info('Shifting y with %s m' % shift_y)
            variables['x'] += shift_x
            variables['y'] += shift_y

        return variables


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
    

    def _get_variables_interpolated_(self, variables, profiles, profiles_depth,
                                     time, reader_x, reader_y, z):

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

        block_before = block_after = None
        blockvariables_before = variables
        blockvars_before = str(variables)
        blockvariables_after = variables
        blockvars_after = str(variables)
        for blockvars in self.var_block_before:
            if all(v in blockvars for v in variables):
                block_before = self.var_block_before[blockvars]
                blockvariables_before = block_before.data_dict.keys()
                blockvars_before = blockvars
                break
            blockvariables_before = variables
            blockvars_before = str(variables)
        for blockvars in self.var_block_after:
            if all(v in blockvars for v in variables):
                block_after = self.var_block_after[blockvars]
                blockvariables_after = block_after.data_dict.keys()
                blockvars_after = blockvars
                break

        # Swap before- and after-blocks if matching times
        if block_before is not None and block_after is not None:
            if block_before.time != time_before:
                if block_after.time == time_before:
                    block_before = block_after
                    self.var_block_before[blockvars_before] = block_before
            if block_after.time != time_after:
                if block_before.time == time_before:
                    block_after = block_before
                    self.var_block_after[blockvars_after] = block_after

        # Fetch data, if no buffer is available
        if block_before is None or \
                block_before.time != time_before:
            reader_data_dict = \
                    self.__convolve_block__(
                self.get_variables(blockvariables_before, time_before,
                                    mx, my, mz)
                )
            self.var_block_before[blockvars_before] = \
                ReaderBlock(reader_data_dict,
                            interpolation_horizontal=self.interpolation)
            try:
                len_z = len(self.var_block_before[blockvars_before].z)
            except:
                len_z = 1
            logger.debug(
                ('Fetched env-block (size %ix%ix%i) ' + 'for time before (%s)')
                % (len(self.var_block_before[blockvars_before].x),
                   len(self.var_block_before[blockvars_before].y), len_z,
                   time_before))
            block_before = self.var_block_before[blockvars_before]
        if block_after is None or block_after.time != time_after:
            if time_after is None:
                self.var_block_after[blockvars_after] = block_before
            else:
                reader_data_dict = self.__convolve_block__(
                    self.get_variables(blockvariables_after, time_after, mx,
                                       my, mz))
                self.var_block_after[blockvars_after] = \
                    ReaderBlock(
                        reader_data_dict,
                        interpolation_horizontal=self.interpolation)
                try:
                    len_z = len(self.var_block_after[blockvars_after].z)
                except:
                    len_z = 1

                logger.debug(('Fetched env-block (size %ix%ix%i) ' +
                              'for time after (%s)') %
                             (len(self.var_block_after[blockvars_after].x),
                              len(self.var_block_after[blockvars_after].y),
                              len_z, time_after))
                block_after = self.var_block_after[blockvars_after]

        if (block_before is not None and block_before.covers_positions(
            reader_x, reader_y) is False) or (\
            block_after is not None and block_after.covers_positions(
                reader_x, reader_y) is False):
            logger.warning('Data block from %s not large enough to '
                           'cover element positions within timestep. '
                           'Buffer size (%s) must be increased. See `Variables.set_buffer_size`.' %
                           (self.name, str(self.buffer)))
            # TODO; could add dynamic incraes of buffer size here

        ############################################################
        # Interpolate before/after blocks onto particles in space
        ############################################################
        self.timer_start('interpolation')
        logger.debug('Interpolating before (%s) in space  (%s)' %
                     (block_before.time, self.interpolation))
        env_before, env_profiles_before = block_before.interpolate(
            reader_x, reader_y, z, variables, profiles, profiles_depth)

        if (time_after is not None) and (time_before != time):
            logger.debug('Interpolating after (%s) in space  (%s)' %
                         (block_after.time, self.interpolation))
            env_after, env_profiles_after = block_after.interpolate(
                reader_x, reader_y, z, variables, profiles, profiles_depth)

        self.timer_end('interpolation')

        #######################
        # Time interpolation
        #######################
        self.timer_start('interpolation_time')
        env_profiles = None
        if (time_after is not None) and (time_before != time) and self.always_valid is False:
            weight_after = ((time - time_before).total_seconds() /
                            (time_after - time_before).total_seconds())
            logger.debug(('Interpolating before (%s, weight %.2f) and'
                          '\n\t\t      after (%s, weight %.2f) in time') %
                         (block_before.time, 1 - weight_after,
                          block_after.time, weight_after))
            env = {}
            for var in variables:
                # Weighting together, and masking invalid entries
                env[var] = np.ma.masked_invalid(
                    (env_before[var] * (1 - weight_after) +
                     env_after[var] * weight_after))
            # Interpolating vertical profiles in time
            if profiles is not None:
                env_profiles = {}
                logger.debug('Interpolating profiles in time')
                # Truncating layers not present both before and after
                numlayers = np.minimum(len(env_profiles_before['z']),
                                       len(env_profiles_after['z']))
                env_profiles['z'] = env_profiles_before['z'][0:numlayers]
                for var in env_profiles_before.keys():
                    if var == 'z':
                        continue
                    env_profiles_before[var] = np.atleast_2d(
                        env_profiles_before[var])
                    env_profiles_after[var] = np.atleast_2d(
                        env_profiles_after[var])
                    env_profiles[var] = (
                        env_profiles_before[var][0:numlayers, :] *
                        (1 - weight_after) +
                        env_profiles_after[var][0:numlayers, :] * weight_after)
            else:
                env_profiles = None

        else:
            logger.debug('No time interpolation needed - right on time.')
            env = env_before
            if profiles is not None:
                if 'env_profiles_before' in locals():
                    env_profiles = env_profiles_before
                else:
                    # Copying data from environment to vertical profiles
                    env_profiles = {'z': profiles_depth}
                    for var in profiles:
                        env_profiles[var] = np.ma.array([env[var], env[var]])
        self.timer_end('interpolation_time')

        return env, env_profiles
    
    
    # these should be moved to physics_methods eventually.
    def apply_logarithmic_current_profile(self,env,z):
        if not self.use_3d and 'sea_floor_depth_below_sea_level' in self.variables and 'x_sea_water_velocity' in self.variables :
            log_profile_factor = self.logarithmic_current_profile(z,env['sea_floor_depth_below_sea_level'])
            logger.debug('Applying logarithmic current profile to 2D current data [x_sea_water_velocity,y_sea_water_velocity] %s <= factor <=%s' % (np.min(log_profile_factor), np.max(log_profile_factor) ))
            env['x_sea_water_velocity'] = log_profile_factor * env['x_sea_water_velocity']
            env['y_sea_water_velocity'] = log_profile_factor * env['y_sea_water_velocity']
            if False:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.plot(z/env['sea_floor_depth_below_sea_level'],log_profile_factor,'.')
                import pdb;pdb.set_trace()
                plt.close()

    def logarithmic_current_profile(self, particle_z, total_depth):
        ''' 
        Extrapolation of depth-averaged currents to any vertical 
        level of the water column assuming a logarithmic profile


        Inputs :
            particle_z : vertical position of particle in water column (negative down as used in Opendrift)
            total_depth : total water depth at particle position (positive down)
            z0 : roughness length, in meters (default, z0 = 0.001m )

        Returns : 
            Factors to be apply to interpolated raw depth-averaged currents

        Reference :
            Van Rijn, 1993. Principles of Sediment Transport in Rivers,
            Estuaries and Coastal Seas

        '''

        # Opendrift convention : particle_z is 0 at the surface, negative down
        # 
        # The particle_z we need is the height of particle above seabed (positive)
        part_z_above_seabed = np.abs(total_depth) + particle_z 
        # note : taking the absolute value enbsure we have positive down depth (though it make any depth<0 wrong..but log profile probably not critical at these points anyway?)
        # if we are sure that total_depth is positive down then we should just use >> part_z_above_seabed = total_depth + particle_z 
        if not hasattr(self,'z0'): 
            self.z0 = 0.001 # typical value for sandy seabed
        log_fac = ( np.log(part_z_above_seabed / self.z0) ) / ( np.log(np.abs(total_depth)/self.z0)-1 ) # total_depth must be positive, hence the abs()
        log_fac[np.where(part_z_above_seabed<=0)] = 1.0 # do not change velocity value
        return log_fac