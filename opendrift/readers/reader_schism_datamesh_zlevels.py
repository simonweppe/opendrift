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
# This reader is based on reader_schism_native.py and allows for 
# ingestion of SCHISM datasets that have been post-processed onto regular
# and fixed z-levels in the vertical (from Oceanum's Datamesh)
# unlike native SCHISM files that have time-varying sigma levels. 
##########################################################################


import logging
logger = logging.getLogger(__name__)

import numpy as np
from datetime import datetime
# from future.utils import iteritems # not needed anymore
from netCDF4 import Dataset, MFDataset, num2date
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree #cython-based KDtree for quick nearest-neighbor search
# consider using https://github.com/storpipfugl/pykdtree for KDtree computations - test efficiency
import pyproj
from opendrift.readers.basereader import BaseReader, UnstructuredReader
from opendrift.readers.basereader import BaseReader, UnstructuredReader
from opendrift.readers.reader_schism_native import Reader as ReaderSchismNative

from opendrift.readers.basereader.consts import *
import xarray as xr
import shapely

class Reader(BaseReader,UnstructuredReader):
# class Reader(ReaderSchismNative):

    def __init__(self, filename=None, name=None, proj4=None ,**kwargs):
        """Initialise reader_netCDF_CF_unstructured_SCHISM

        Args:
            filename    :   name of SCHISM netcdf file (can have wildcards)

            name        :   name of reader - optional, taken as filename if not input
                            o.readers['name']

            proj4       :   proj4 string defining spatial reference system to use to convert the (lon,lat) from netcdd file
                            to cartesian coordinates. This is required for correct use of the 3D KDtree (i.e. all distances in meters)
                            to convert 3D velocities to particle positions
                            find string here : https://spatialreference.org/ref/epsg/
                                        
            kwargs      : None for now
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
        
        # [name_used_in_schism : equivalent_CF_name] h
        # here we updated with names used in processed files
        schism_mapping = {
            'hvelu': 'x_sea_water_velocity', 
            'hvelv': 'y_sea_water_velocity',
            'temp' : 'sea_water_temperature',
            'salt' : 'sea_water_salinity'}
            # 'hvel': 'x_sea_water_velocity',
            # 'hvel': 'y_sea_water_velocity',
            # 'depth': 'sea_floor_depth_below_sea_level',
            # 'elev' : 'sea_surface_height',

            # 'zcor' : 'vertical_levels', # time-varying vertical coordinates
            # 'sigma': 'ocean_s_coordinate',
            # 'vertical_velocity' : 'upward_sea_water_velocity',
            # 'wetdry_elem': 'land_binary_mask',
            # 'wind_speed' : 'x_wind',
            # 'wind_speed' : 'y_wind' 
            # diffusivity
            # viscosity

        self.return_block = True

        try:
            # Open file, check that everything is ok
            logger.info('Opening dataset: ' + filestr)
            if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
                logger.info('Opening files with open_mfdataset')
                self.dataset = xr.open_mfdataset(filename,chunks={'time': 1}).drop_duplicates(dim = 'time', keep='last')
                # in case of issues with file ordering consider inputting an explicit filelist 
                # to reader, for example:
                # 
                # ordered_filelist_1 = []
                # ordered_filelist_1.extend(glob.glob(data_path + 'schism_marl2008*_00z_3D.nc'))# month block 1
                # ordered_filelist_1.sort()
            else:
                logger.info('Opening file with dataset')
                self.dataset = xr.open_dataset(filename,chunks={'time': 1})

        except Exception as e:
            raise ValueError(e)

        # Define projection to be used to convert (lon,lat) from netcdf file
        # to cartesian coordinates
        # self.proj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' #'+proj=latlong' # WGS84
        # 
        if proj4 is not None: #  user has provided a projection apriori
            self.proj4 = proj4
        else: # no input for proj4
            self.proj4 = None
            logger.error('No projection <proj4> was defined when initializing the reader')
            logger.error('Please specify cartesian <proj4> to be used to convert model outputs coordinates \n \
                         (needed for 3D kdtree to work i.e. to convert all distances in meters')
        
        self.use_3d = True # we always use 3D here

        if self.use_3d and ('hvelu' not in self.dataset.variables or 'hvelv' not in self.dataset.variables):
            logger.error('No 3D velocity data in file - cannot find variable ''hvelu, or hvelv'' ')
        
        # we fill the nan data in the vertical with 0.0
        # particle will be flagged as below seabed anyway
        logger.info('Filling nan with latest 0.0 values in 3D velocity arrays')
        self.dataset['hvelu'] = self.dataset.hvelu.fillna(0)
        self.dataset['hvelv'] = self.dataset.hvelv.fillna(0)

        # Alternative could bbe to fill the nan data in the vertical with latest good values
        # logger.info('Filling nan with latest non-nan values in 3D velocity arrays')
        # self.dataset['hvelu'] = self.dataset.hvelu.ffill(dim='lev',limit=None)
        # self.dataset['hvelv'] = self.dataset.hvelv.ffill(dim='lev',limit=None)

        logger.debug('Finding coordinate variables.')
        # Find x, y and z coordinates
        for var_name in self.dataset.variables:

            var = self.dataset.variables[var_name]

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
                if var.ndim == 2:
                    # When datasets are concatenated by mfdataset(), coordinates vector (1D) may
                    # be tiled to a 2D array of size (time,node), keep only one vector for x,y  
                    var = var[0,:]
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
                if var.ndim == 2:
                    # When datasets are concatenated by mfdataset(), coordinates vector (1D) may
                    # be tiled to a 2D array of size (time,node), keep only one vector for x,y  
                    var = var[0,:]
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    unitfactor = 1000
                else:
                    unitfactor = 1
                var_data = var.values
                y = var_data*unitfactor
                self.numy = var.shape[0]
            if standard_name == 'depth' or axis == 'Z':
                var_data = var.values
                if 'positive' not in var.attrs or \
                        var.attrs['positive'] == 'up':
                    self.z = var_data
                else:
                    self.z = -var_data
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

        if 'x' not in locals():
            raise ValueError('Did not find x-coordinate variable')
        if 'y' not in locals():
            raise ValueError('Did not find y-coordinate variable')

        self.x = x
        self.y = y

        if not (self.x>360.).any() and self.use_3d :
            logger.debug('Native coordinates in SCHISM outputs (SCHISM_hgrid_node_x,SCHISM_hgrid_node_y) are in lon/lat (WGS84)')
            logger.debug('Converting to user-defined proj4 defined when initialising reader : %s' % self.proj4)
            # The 3D interpolation doesnt work directly if the x,y coordinates in native netcdf files
            # are not cartesian but geographic. In that case, when doing 3D interpolation and tree search, 
            # the vertical distance unit is meter, while the horizontal distance unit is degrees, which will return
            # erroneous "closest" nodes in ReaderBlockUnstruct,interpolate()
            # 
            if self.proj4 is None :
                logger.error('No projection <proj4> was specified when initializing reader')
                logger.error('If native coordinates are lon/lat, then specify the coords system to be used to convert these to cartesian')

            # convert lon/lat to user-defined cartesian coordinate system
            proj_wgs84 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # proj4 string for WGS84
            transformer = pyproj.Transformer.from_proj(proj_from = proj_wgs84, proj_to = self.proj4,always_xy = True)
            x2, y2 = transformer.transform(self.x, self.y)
            self.x = x2.copy()
            self.y = y2.copy()
        
        # Run constructor of parent Reader class
        super(Reader, self).__init__()

        # compute CKDtree of (static) 2D nodes using _build_ckdtree_() from unstructured.py
        logger.debug('Building CKDtree of static 2D nodes for nearest-neighbor search')
        self.reader_KDtree = self._build_ckdtree_(self.x,self.y) 

        # compute CKDtree of (static) 2D nodes using _build_ckdtree_() from unstructured.py
        # also output tiled version of X,Y,Z 
        logger.debug('Building CKDtree of static 3D nodes for nearest-neighbor search')
        self.reader_KDtree = self.build_3d_kdtree()
        logger.debug('Building CKDtree of static 3D nodes - %s zlevels : %s' % (len(np.unique(self.reader_KDtree.data[:,2])),np.unique(self.reader_KDtree.data[:,2]))) 

        # build convex hull of points for particle-in-mesh checks using _build_boundary_polygon_() from unstructured.py
        logger.debug('Building convex hull of nodes for particle''s in-mesh checks')
        self.boundary = self._build_boundary_polygon_(self.x,self.y)
        
        # Find all variables having standard_name
        self.variable_mapping = {}
        for var_name in self.dataset.variables:
            if var_name in [self.xname, self.yname]: #'depth'
                continue  # Skip coordinate variables
            var = self.dataset.variables[var_name]
            attributes = var.attrs
            att_dict = var.attrs

            if var_name in schism_mapping:
                self.variable_mapping[schism_mapping[var_name]] = str(var_name) 
                    
        self.variables = list(self.variable_mapping.keys())

        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        # self.xmin = self.lon.min()
        # self.xmax = self.lon.max()
        # self.ymin = self.lat.min()
        # self.ymax = self.lat.max()

        # Run constructor of parent Reader class
        super(Reader, self).__init__()
        
        # Dictionaries to store blocks of data for reuse (buffering)
        self.var_block_before = {}  # Data for last timestep before present
        self.var_block_after = {}   # Data for first timestep after present
  
    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None, block=False):

        """ The function extracts 'requested_variables' from the interpolated zlevels SCHISM files
            which will then be used in _get_variables_interpolated_() to initialise the ReaderBlockUnstruct objects 
            used to interpolate data in space and time

            For now the function will extract the entire slice of data of 'requested_variables' at given 'time'

            There is an option to extract only a subset of data around particles clouds to have less data but
            it means we need to recompute the KDtree of the subset nodes every time in ReaderBlockUnstruct.
            
            Speed gain to be tested ...
        """
        requested_variables, time, x, y, z, outside = \
            self.check_arguments(requested_variables, time, x, y, z)

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        variables = {'x': self.x, 'y': self.y, 'z': 0.*self.y,
                     'time': nearestTime}
        
        # extracts the full slices of requested_variables at time indxTime
        for par in requested_variables:
            # if par not in ['x_sea_water_velocity','y_sea_water_velocity','land_binary_mask','x_wind','y_wind'] :
            # standard case - for all variables except vectors such as current, wind, etc..
            var = self.dataset.variables[self.variable_mapping[par]]
            if var.ndim == 1:
                data = var[:] # e.g. depth
                logger.debug('reading constant data from unstructured reader %s' % (par))
            elif var.ndim == 2: 
                data = var[indxTime,:] # e.g. 2D temperature
                logger.debug('reading 2D data from unstructured reader %s' % (par))
            elif var.ndim == 3:
                data = var[indxTime,:,:] # e.g. 3D salt [time,node,lev]
                logger.debug('reading 3D data from unstructured reader %s' % (par))
                # convert 3D data matrix to one column array and define corresponding data coordinates [x,y,z]
                # (+ update variables dictionary with 3d coords if needed)
                data,variables = self.convert_3d_to_array(indxTime,data,variables)
            else:
                raise ValueError('Wrong dimension of %s: %i' %
                                    (self.variable_mapping[par], var.ndim))
            
            variables[par] = data # save all data slice to dictionary with key 'par'
            # Store coordinates of returned points
            variables[par] = np.asarray(variables[par])

        return variables 


    def convert_3d_to_array(self,id_time,data,variable_dict):
            ''' 
            The function reshapes a data matrix of dimensions = [node,vertical_levels] (i.e. data at vertical levels, at given time step) 
            into a one-column array and works out corresponding 3d coordinates [lon,lat,z] using the 
            fixed Zlevels. (reader_native_schism.py uses the time-varying 'zcor' variable instead)

            args:
                -id_time
                -data
                -variable_dict
            out :
                -flattened 'data' array
                -addition of ['x_3d','y_3d','z_3d'] items to variable_dict if needed.
            '''
                        

            try:
                # vertical_levels = self.dataset.variables['zcor'][id_time,:,:]

                # depth are negative down consistent with convention used in OpenDrift 
                # if using the netCDF4 library, vertical_levels is masked array where "masked" levels are those below seabed  (= 9.9692100e+36)
                # if using the xarray library, vertical_levels is nan for levels are those below seabed

                # convert to masked array to be consistent with what netCDF4 lib returns
                # vertical_levels = np.ma.array(vertical_levels, mask = np.isnan(vertical_levels.data)) 

                # Z levels are constant over time here
                vertical_levels = np.ma.array(self.reader_KDtree.data[:,2], mask = np.isnan(self.reader_KDtree.data[:,2])) 
                data = np.asarray(data)
                # vertical_levels.mask = np.isnan(vertical_levels.data) # masked using nan's when using xarray
                # flatten 3D data 
                data = np.ravel(data)
                # data = np.ravel(data[~vertical_levels.mask])
            except:
                logger.debug('no vertical level information present in file ... stopping')
                import pdb;pdb.set_trace()
            
            # add corresponding 3D coordinates to the 'variable_dict' which is eventually passed to get_variables_interpolated()
            # They are saved as ['x_3d','y_3d','z_3d'] rather than ['x','y','z'] as it would break things when both 2d and 3d data are requested.
            if 'z_3d' not in variable_dict.keys():
                variable_dict['x_3d'] = np.ravel(self.reader_KDtree.data[:,0]) 
                variable_dict['y_3d'] = np.ravel(self.reader_KDtree.data[:,1])
                variable_dict['z_3d'] = np.ravel(self.reader_KDtree.data[:,2])
        
            return data,variable_dict

    def build_3d_kdtree(self):
        ''' 
        build the static 3D KDtree

        we tile to (x,y) coords to match the nb_lev, then ravel() all to a single 3-column array [x,y,z]

        returns
            - 3D KDTree
            - Zlevels matrix sisze [nnode,nb_lev]
        '''
        zlevels = self.dataset.lev.values
        nb_lev  = self.dataset.dims['lev']
        nb_node =  self.dataset.dims['nSCHISM_hgrid_node']
        # now we need to tile the (x,y) coordinates <nb_lev> time
        # and zlevels <nb_node> times, then ravel() all
        XX = np.tile(self.x,(nb_lev,1)).T
        YY = np.tile(self.y,(nb_lev,1)).T
        ZZ = np.tile(zlevels.T,(nb_node,1))
        # check tiling is consistent shape [nSCHISM_hgrid_node,lev]
        # (XX[0,:]== XX[0,0]).all() 
        # (YY[0,:]== YY[0,0]).all()
        # (ZZ[:,0]== ZZ[0,0]).all()
        
        return cKDTree(np.vstack((XX.ravel(),YY.ravel(),ZZ.ravel())).T)
                
    def set_convolution_kernel(self, convolve):
        """Set a convolution kernel or kernel size (of array of ones) used by `get_variables` on read variables."""
        self.convolve = convolve

    def __convolve_block__(self, env):
        """
        Convolve arrays with a kernel, if reader.convolve is set
        """
        if self.convolve is not None:
            from scipy import ndimage
            N = self.convolve
            if isinstance(N, (int, np.integer)):
                kernel = np.ones((N, N))
                kernel = kernel / kernel.sum()
            else:
                kernel = N
            logger.debug('Convolving variables with kernel: %s' % kernel)
            for variable in env:
                if variable in ['x', 'y', 'z', 'time']:
                    pass
                else:
                    if env[variable].ndim == 2:
                        env[variable] = ndimage.convolve(env[variable],
                                                         kernel,
                                                         mode='nearest')
                    elif env[variable].ndim == 3:
                        env[variable] = ndimage.convolve(env[variable],
                                                         kernel[:, :, None],
                                                         mode='nearest')
        return env

    def _get_variables_interpolated_(self, variables, profiles,
                                   profiles_depth, time,
                                   reader_x, reader_y, z):

        """
        This method _must_ be implemented by every reader. Usually by
        subclassing one of the reader types (e.g.
        :class:`structured.StructuredReader`).

        Arguments are in _native projection_ of reader.

        .. seealso:

            * :meth:`get_variables_interpolated_xy`.
            * :meth:`get_variables_interpolated`.
        """

        """
           Here, this function overloads the _get_variables_interpolated_() methods
           available in unstructured.py.  (which currently doesnt make use of blocks)

           The _get_variables_interpolated_() from structured.py uses regularly gridded 
           data "ReaderBlock" extracted from the netcdf files (which may possibly be "cached" 
           for speed improvements - see code for more detail).

           This function follows a similar approach but is instead using the native 
           high-resolution SCHISM data stored in "ReaderBlockUnstruct" which are used to 
           interpolate data in space and time. 

           The function returns environment data 'env' interpolated at particle positions [x,y] 

        """
        # block = False # legacy stuff 

        # Find reader time_before/time_after
        time_nearest, time_before, time_after, i1, i2, i3 = \
            self.nearest_time(time)
        logger.debug('Reader time:\n\t\t%s (before)\n\t\t%s (after)' %
                      (time_before, time_after))
        # For variables which are not time dependent, we do not care about time
        static_variables = ['sea_floor_depth_below_sea_level', 'land_binary_mask']

        if time == time_before or all(v in static_variables for v in variables):
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

        # z = z.copy()[ind_covered]  # Send values and not reference
        # EDIT:commented out: we need to keep the full array so that shapes are consistent
        
        # start interpolation procedure
        # 
        # general idea is to create a  new "ReaderBlockUnstruct" class that will be called instead of
        # the regular "ReaderBlock" as in basereader.py
        # This allows re-using almost the same code as structured.py for the block_before/block_after
        
        self.timer_start('preparing')        
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
                                    mx, my, mz))          
            # now use reader_data_dict to initialize 
            # a ReaderBlockUnstruct
            logger.debug('initialize ReaderBlockUnstruct var_block_before')
            self.var_block_before[blockvars_before] = \
                ReaderBlockUnstruct(reader_data_dict,
                    KDtree = self.reader_KDtree,
                    interpolation_horizontal=self.interpolation)
            try:
                len_z = len(self.var_block_before[blockvars_before].z)
            except:
                len_z = 1
            logger.debug(('Fetched env-block (size %ix%ix%i) ' +
                          'for time before (%s)') %
                          (len(self.var_block_before[blockvars_before].x),
                           len(self.var_block_before[blockvars_before].y),
                           len_z, time_before))
            block_before = self.var_block_before[blockvars_before]
        if block_after is None or block_after.time != time_after:
            if time_after is None:
                self.var_block_after[blockvars_after] = \
                    block_before
            else:
                reader_data_dict = self.__convolve_block__(
                    self.get_variables(blockvariables_after, time_after, mx,
                                       my, mz))
                self.timer_start('preparing')
                logger.debug('initialize ReaderBlockUnstruct var_block_after')
                self.var_block_after[blockvars_after] = \
                    ReaderBlockUnstruct(
                        reader_data_dict,
                        KDtree = self.reader_KDtree,
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
            import pdb;pdb.set_trace()
            logger.warning('Data block from %s not large enough to '
                            'cover element positions within timestep. '
                            'Buffer size (%s) must be increased.' %
                            (self.name, str(self.buffer)))
        self.timer_end('preparing') 
        ############################################################
        # Interpolate before/after blocks onto particles in space
        ############################################################
        self.timer_start('interpolation')
        logger.debug('Interpolating before (%s) in space  (%s)' %
                      (block_before.time, self.interpolation))
        env_before, env_profiles_before = block_before.interpolate(
                reader_x, reader_y, z, variables,
                profiles, profiles_depth)

        if (time_after is not None) and (time_before != time):
            logger.debug('Interpolating after (%s) in space  (%s)' %
                          (block_after.time, self.interpolation))
            env_after, env_profiles_after = block_after.interpolate(
                    reader_x, reader_y, z, variables,
                    profiles, profiles_depth)

        self.timer_end('interpolation')
        #######################
        # Time interpolation
        #######################
        self.timer_start('interpolation_time')
        env_profiles = None
        if (time_after is not None) and (time_before != time) and self.return_block is True:
            weight_after = ((time - time_before).total_seconds() /
                            (time_after - time_before).total_seconds())
            logger.debug(('Interpolating before (%s, weight %.2f) and'
                           '\n\t\t      after (%s, weight %.2f) in time') %
                          (block_before.time, 1 - weight_after,
                           block_after.time, weight_after))
            env = {}
            for var in variables:
                # Weighting together, and masking invalid entries
                env[var] = np.ma.masked_invalid((env_before[var] *
                                                (1 - weight_after) +
                                                env_after[var] * weight_after))

                if var in standard_names.keys():
                    invalid = np.where((env[var] < standard_names[var]['valid_min'])
                               | (env[var] > standard_names[var]['valid_max']))[0]
                    if len(invalid) > 0:
                        logger.warning('Invalid values found for ' + var)
                        logger.warning(env[var][invalid])
                        logger.warning('(allowed range: [%s, %s])' %
                                        (standard_names[var]['valid_min'],
                                         standard_names[var]['valid_max']))
                        logger.warning('Replacing with NaN')
                        env[var][invalid] = np.nan
            # Interpolating vertical profiles in time
            if profiles is not None:
                env_profiles = {}
                logger.info('Interpolating profiles in time')
                # Truncating layers not present both before and after
                numlayers = np.minimum(len(env_profiles_before['z']),
                                       len(env_profiles_after['z']))
                env_profiles['z'] = env_profiles_before['z'][0:numlayers+1]
                for var in env_profiles_before.keys():
                    if var == 'z':
                        continue
                    env_profiles_before[var]=np.atleast_2d(env_profiles_before[var])
                    env_profiles_after[var]=np.atleast_2d(env_profiles_after[var])
                    env_profiles[var] = (
                        env_profiles_before[var][0:numlayers, :] *
                        (1 - weight_after) +
                        env_profiles_after[var][0:numlayers, :]*weight_after)
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
        # the masking, rotation etc.. is done in variables.py get_variables_interpolated_xy()

        # apply log profile if we are interpolating 2D data for ['x_sea_water_velocity','y_sea_water_velocity']
        # not using for now
        if False :
            self.apply_logarithmic_current_profile(env,z)
         
        # make sure dry points have zero velocities which is not always the case
        # we could also look at using depth and thresholds to flag other dry points ?
        if 'land_binary_mask' in env.keys():
            logger.debug('Setting [x_sea_water_velocity,y_sea_water_velocity] to zero at dry points')
            env['x_sea_water_velocity'][env['land_binary_mask'].astype('bool')] = 0
            env['y_sea_water_velocity'][env['land_binary_mask'].astype('bool')] = 0
        
        return env, env_profiles

    def covers_positions_xy(self, x, y, z=0):
        """
        Check which points are within boundary of mesh.

        Wrapper function of covers_positions() from unstructured.py which is called in 
        get_variables_interpolated_xy() function from variables.py 

        It returns indices of in-mesh points, and in-mesh point coordinates rather than a boolean array (inside/outside) 

        Within get_variables_interpolated_xy() from variables.py, data is queried for these in-mesh points only and the 
        full array (incl. out of mesh positions) is re-generated with correct masking 

        """
        ind_covered = np.where(self.covers_positions(x, y))[0]
        return ind_covered ,x[ind_covered], y[ind_covered]

###########################
# ReaderBlockUnstruct class
###########################

# horizontal_interpolation_methods = {
#     'nearest': Nearest2DInterpolator,
#     'ndimage': NDImage2DInterpolator,
#     'linearND': LinearND2DInterpolator,
#     'linearNDFast': Linear2DInterpolator}


# vertical_interpolation_methods = {
#     'nearest': Nearest1DInterpolator,
#     'linear': Linear1DInterpolator}


class ReaderBlockUnstruct():
    """Class to store and interpolate the data from an *unstructured* reader.
       This is the equivalent of ReaderBlock (regular grid) for *unstructured* grids.

       arguments: (in addition to ReaderBlock)

           KDtree : for nearest-neighbor search (initialized using SCHISM nodes in reader's _init_() )
                    This is read from reader object, so that it is not recomputed every time

    """
    logger = logging.getLogger('opendrift')  # using common logger

    def __init__(self, data_dict, 
                 KDtree = None, # here it should be the 3D KDtree
                 interpolation_horizontal='linearNDFast',
                 interpolation_vertical='linear'):

        # Make pointers to data values, for convenience
        self.x = data_dict['x']
        self.y = data_dict['y']
        self.time = data_dict['time']
        self.data_dict = data_dict
        del self.data_dict['x']
        del self.data_dict['y']
        del self.data_dict['time']
        try:
            self.z = data_dict['z'] 
            # probably not valid ..since z are different for each point in SCHISM..rather than fixed
            # This is used for the profile interpolation that is not yet functional 
            #
            del self.data_dict['z']
        except:
            self.z = None    
        # if some 3d data is provided, save additional dict entries
        if 'z_3d' in data_dict.keys():
            self.x_3d = data_dict['x_3d']
            self.y_3d = data_dict['y_3d']
            self.z_3d = data_dict['z_3d'] 
            del self.data_dict['x_3d']
            del self.data_dict['y_3d']
            del self.data_dict['z_3d']      

        # Initialize KDtree(s) 
        # > save the 2D one by default (initizalied during reader __init__()
        # > compute and save the time-varying 3D KDtree if relevant  
        
        logger.debug('saving reader''s 2D (horizontal) KDtree to ReaderBlockUnstruct')
        self.block_KDtree = KDtree # KDtree input to function = one computed during reader's __init__()
        
        # the KDtree passed to interpolator is actually a 3D KDtree, and it will be static now
        logger.debug('saving reader''s 3D KDtree (static) to ReaderBlockUnstruct')
        self.block_KDtree_3d  = KDtree

        if 'land_binary_mask' in self.data_dict.keys() and \
                interpolation_horizontal != 'nearest':
            logger.debug('Nearest interpolation will be used '
                          'for landmask, and %s for other variables'
                          % interpolation_horizontal)

    def _initialize_interpolator(self, x, y, z=None):
        logger.debug('Initialising interpolator.')
        self.interpolator2d = self.Interpolator2DClass(self.x, self.y, x, y)
        if self.z is not None and len(np.atleast_1d(self.z)) > 1:
            self.interpolator1d = self.Interpolator1DClass(self.z, z)

    def interpolate(self, x, y, z=None, variables=None,
                    profiles=[], profiles_depth=None):
        # Use the KDtree to interpolate data to [x,y,z] particle positions

        # self._initialize_interpolator(x, y, z)
        
        env_dict = {}
        if profiles is not []:
            # profiles_dict = {'z': self.z} # probably not valid...
            profiles_dict = {'z': profiles_depth} # consistent with what is done in <unstructured.py> line 70
        for varname, data in self.data_dict.items(): # same syntax as in structured.py, used to be iteritems(self.data_dict)
            nearest = False
            # land mask 
            if varname == 'land_binary_mask':
                # here we need to make a choice on when to flag as land 
                if False :
                    nearest = True
                    self.interpolator2d_nearest = Nearest2DInterpolator(self.x, self.y, x, y)
                # if closest node is dry then we assume particle is on land
                nb_closest_nodes = 1
                #2D KDtree
                dist,i=self.block_KDtree.query(np.vstack((x,y)).T,nb_closest_nodes, workers=-1) #quick nearest-neighbor lookup
                data_interpolated = data[i] # we keep closest value

            # ensemble data
            if type(data) is list:
                num_ensembles = len(data)
                logger.debug('Interpolating %i ensembles for %s' % (num_ensembles, varname))
                if data[0].ndim == 2:
                    horizontal = np.zeros(x.shape)*np.nan
                else:
                    horizontal = np.zeros((len(self.z), len(x)))*np.nan
                ensemble_number = np.remainder(range(len(x)), num_ensembles)
                for en in range(num_ensembles):
                    elnum = ensemble_number == en
                    int_full = self._interpolate_horizontal_layers(data[en], nearest=nearest)
                    if int_full.ndim == 1:
                        horizontal[elnum] = int_full[elnum]
                    else:
                        horizontal[:, elnum] = int_full[:, elnum]
            # standard data 2D or 3D
            else:
                # use KDtree to find nearest neighbours and interpolate based on distance, on 2D or 3D
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html#scipy.spatial.cKDTree.query
                # print(varname)

                nb_closest_nodes = 3
                DMIN=1.e-10
                # Note : we always use the 3D KDtree even if there is actually only vertical level in the SCHISM z-levels files
                if False: # if data.shape[0] == self.x.shape[0] : # 2D data- full slice
                    #2D KDtree
                    dist,i=self.block_KDtree.query(np.vstack((x,y)).T,nb_closest_nodes, workers=-1) #quick nearest-neighbor lookup
                    # dist = distance to nodes / i = index of nodes
                elif hasattr(self,'z_3d') and (data.shape[0] == self.x_3d.shape[0]) : #3D data
                    #3D KDtree
                    dist,i=self.block_KDtree_3d.query(np.vstack((x,y,z)).T,nb_closest_nodes, workers=-1) #quick nearest-neighbor lookup
                    # dist = distance to nodes / i = index of nodes
                    ##############################
                    # PLOT CHECKS
                    if False:
                        import matplotlib.pyplot as plt;plt.ion();plt.show()
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(self.x_3d[i[0]].data.tolist(),self.y_3d[i[0]].data.tolist(),self.z_3d[i[0]].data.tolist(),c='r', marker='o')
                        ax.scatter(x[0:1],y[0:1],z[0:1],c='g', marker='o')
                    ##############################3
                
                dist[dist<DMIN]=DMIN
                fac=(1./dist)
                data_interpolated = (fac*data.take(i)).sum(-1)/fac.sum(-1)


                # # CHECK#################################################
                # data_KD = np.vstack((self.x_3d,self.y_3d,self.z_3d)).T
                # data_points_to_find = np.vstack((x,y,z)).T
                # data_KD[i[0][0],:]
                # data_points_to_find[0]
                # ########################################################
 
                # horizontal = self._interpolate_horizontal_layers(data, nearest=nearest)
            
            if profiles is not None and varname in profiles:
                # not really functional yet...we should interpolate data at top and bottom of profiles here,
                profiles_dict[varname] = data_interpolated # horizontal

            # if horizontal.ndim > 1:
            #     env_dict[varname] = self.interpolator1d(data_interpolated) #self.interpolator1d(horizontal)
            # else:

            env_dict[varname] = data_interpolated #horizontal

        if 'z' in profiles_dict:
            profiles_dict['z'] = np.atleast_1d(profiles_dict['z'])

        return env_dict, profiles_dict

    def _interpolate_horizontal_layers(self, data, nearest=False):
        '''Interpolate all layers of 3d (or 2d) array.'''
    
        if nearest is True:
            interpolator2d = self.interpolator2d_nearest
        else:
            interpolator2d = self.interpolator2d
        if data.ndim == 2:
            return interpolator2d(data)
        if data.ndim == 3:
            num_layers = data.shape[0]
            # Allocate output array
            result = np.ma.empty((num_layers, len(interpolator2d.x)))
            for layer in range(num_layers):
                result[layer, :] = self.interpolator2d(data[layer, :, :])
            return result


    def covers_positions(self, x, y, z=None):
        '''Check if given positions are covered by this reader block.'''
        
        indices = np.where((x >= self.x.min()) & (x <= self.x.max()) &
                           (y >= self.y.min()) & (y <= self.y.max()))[0]

        if len(indices) != len(x):
            import matplotlib.pyplot as plt
            plt.ion()
            plt.plot(x,y,'r.')
            box = np.array([(self.x.min(),self.y.min()),\
            (self.x.max(),self.y.min()),\
            (self.x.max(),self.y.max()),\
            (self.x.min(),self.y.max()),\
            (self.x.min(),self.y.min())])
            plt.plot(box[:,0],box[:,1],'k--')
            plt.title('Increase buffer distance around particle cloud')
            import pdb;pdb.set_trace()
            plt.close()

        if len(indices) == len(x):
            return True
        else:
            return False


    # def covers_positions(self, lon, lat, z=0):
    #     """Return indices of input points covered by reader.
        
    #     For an unstructured reader, this is done by checking that if particle positions are within the convex hull
    #     of the mesh nodes. This means it is NOT using the true polygon bounding the mesh ..but this is generally good enough
    #     to locate the outer boundary (which is often semi-circular). 
    #     >> The convex hull will not be good for the shorelines though, but this is not critical since coast interaction 
    #     will be handled by either by the landmask (read from file or using global_landmask) 
    #     Data interpolation might be an issue for particles reaching the land, and not actually flagged as out-of-bounds...
    #     To Check...

    #     Better alternatives could be: 
    #         - get the true mesh polygon from netCDF files..doesnt seem to be available.
    #         - compute an alpha-shape of mesh nodes (i.e. enveloppe) and use that as polygon. better than convexhull but not perfect 
    #                 (shp = alphaShape(double(x),double(y),10000); work well in matlab for example)
    #         - workout that outer mesh polygon from a cloud of points...maybe using a more evolved trimesh package ?

    #     """        
    #     # weird bug in which x,y become 1e30 sometimes...
    #     # need to print stuff to make it work...

    #     # Calculate x,y coordinates from lon,lat
    #     print(lon.max()) # if I comment this...[x,y] may become 1e+30..have no idea why
    #                      # it runs fine if I have that print statement ....
    #     # print(lat.max())
    #     x, y = self.lonlat2xy(lon, lat)      
    #     # Only checking vertical coverage if zmin, zmax is defined
    #     zmin = -np.inf
    #     zmax = np.inf
    #     if hasattr(self, 'zmin') and self.zmin is not None:
    #         zmin = self.zmin
    #     if hasattr(self, 'zmax') and self.zmax is not None:
    #         zmax = self.zmax

    #     # sometimes x,y will are = 1e30 at this stage..and have no idea why
    #     # recomputing them below seems to fix the issue        
    #     # x, y = self.lonlat2xy(lon, lat)

    #     if self.global_coverage():
    #         pass
    #         # unlikely to be used for SCHISM domain
    #     else:
    #         # Option 1 : use the Path object (matplotlib) defined in __init__() : self.hull_path
    #         # in_hull =self.hull_path.contains_points(np.vstack([x,y]).T)
    #         # Option 2 : use the Prepared Polygon object
    #         in_hull = UnstructuredReader.covers_positions(self,x, y, z)
    #         indices = np.where(in_hull) 
    #     try:
    #         return indices, x[indices], y[indices]
    #     except:
    #         return indices, x, y    