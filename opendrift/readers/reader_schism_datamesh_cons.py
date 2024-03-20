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
from opendrift.readers.basereader import BaseReader, UnstructuredReader
from opendrift.readers.basereader.consts import *
import xarray as xr
import shapely
import oceantide

class Reader(BaseReader,UnstructuredReader):

    def __init__(self, filename=None, name=None, use_mesh_polygon = True,**kwargs):
        """Initialise reader_netCDF_CF_unstructured_SCHISM

        Args:
            filename    :   name of unstructured constituent grid from Oceanum's Datamesh

            name        :   name of reader - optional, taken as filename if not input
                            o.readers['name']
            
            use_mesh_polygon : Switch to use the mesh polygon saved in constituent grid file True by default

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
            'h' : 'sea_surface_height',
            'dep': 'land_binary_mask',}

        self.return_block = True

        try:
            # Open file, check that everything is ok
            logger.info('Opening dataset: ' + filestr)
            if ('nc' not in filestr) :
                logger.info('Opening files with open_zarr')
                self.dataset = xr.open_zarr(filestr)
            else:
                logger.info('Opening file with open_dataset')
                self.dataset = xr.open_dataset(filestr,chunks={'time': 1})
            # need to edit the cons name for correct use in oceantide later on
            self.dataset['con']=[x.strip().upper() for x in self.dataset['cons'].values]

        except Exception as e:
            raise ValueError(e)

        # Define projection of input data - will always be lon/lat
        self.proj4 = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' #'+proj=latlong'
        
        # use dummy start/end times instead, to make it always valid time-wise
        self.start_time = datetime(1000,1,1) 
        self.end_time = datetime(3000,1,1) 
        
        logger.debug('Finding coordinate variables.')
        # Find x, y and z coordinates
        for var_name in self.dataset.variables:

            if var_name in ['boundary','island','elements','cons']:
                continue

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

                if var.ndim == 2: # should not happen in theory but..
                    # when datasets are concatenated by mfdataset(), coordinates vector (1D) may
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
                if var.ndim == 2: #  should not happen in theory but..
                    # when datasets are concatenated by mfdataset(), coordinates vector (1D) may
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
        
        # Run constructor of parent Reader class
        super(Reader, self).__init__()

        # compute CKDtree of (static) 2D nodes using _build_ckdtree_() from unstructured.py
        logger.debug('Building CKDtree of static 2D nodes for nearest-neighbor search')
        self.reader_KDtree = self._build_ckdtree_(self.x,self.y) 

        # build convex hull of points for particle-in-mesh checks using _build_boundary_polygon_() from unstructured.py
        logger.debug('Building mesh boundary and interior islands for in-mesh checks')
        self.boundary,self.boundary_with_islands  = self._build_boundary_polygon_(self.x,self.y)

        self.use_mesh_polygon = use_mesh_polygon
        if self.use_mesh_polygon : 
            logger.debug('Using mesh polygon saved in constituent grid for on-land particles checks')
            self.mesh_polygon =  self.boundary_with_islands # prepared geometry to be used for in-poly checks

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

        # Run constructor of parent Reader class
        super(Reader, self).__init__()
        
        # Dictionaries to store blocks of data for reuse (buffering)
        self.var_block_before = {}  # Data for last timestep before present
        self.var_block_after = {}   # Data for first timestep after present
        
    def build_ckdtree(self,x,y):
        # This is done using cython-based cKDTree from scipy for quick nearest-neighbor search
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
        # self.reader_KDtree = cKDTree(np.vstack((self.lon,self.lat)).T) 
        return cKDTree(np.vstack((x,y)).T) 
    
    def _build_boundary_polygon_(self, x, y):
        """
        The methods builds 2 polygons to be used :
            - to check if particles are within mesh outer boundary (to decide if we used this dataset as driver) 
            - to check if particles are within mesh and not on interior islands (defined as holes)
        
        This version overloads version in unstructured.py

        We use the boundary information saved in the cons file
        to build the geometry instead of using an approximation 
        with the convex hull as in reader_schism_native.py

        The outer boundary mesh polygon will be used in covers_positions() while the polygons with islands
        will be used for coastlines intersection checks.

        To double check > will a particle become "beached" if it crosses the mesh open boundary ? 
        or will the covers_positions will correctly flag it as out-of-bounds ?

        Arguments:
            :param x: Array of node x position, lenght N
            :param y: Array of node y position, length N

        Returns:
            Two `shapely.prepared.prep` `shapely.Polygon`.

            A polygon defining the outer boundary of the mesh only, and
            a polygon defining the outer boundary of the mesh AND including holes for islands.

        """
        from shapely.geometry import Polygon
        from shapely.prepared import prep
        from scipy.spatial import ConvexHull

        outer_bnd_id = np.int64(self.dataset.boundary) # indices of mesh boundary
        mesh_poly = np.vstack((x[outer_bnd_id], y[outer_bnd_id])).T

        # now generate a list of island coords to be specified as holes in the mesh boundary
        # island_polys = [np.int64(island.isel(inum=ii).dropna(dim='inode')).tolist() for ii in island.inum] # wont work..
        island_polys = []
        for ii in self.dataset.inum:
            id_island_i = np.int64(self.dataset.island.isel(inum=ii).dropna(dim='inode'))
            poly_i = np.vstack((x[id_island_i],y[id_island_i])).T
            island_polys.append(poly_i)

        # make some prepared geometries for in-polys checks
        boundary = Polygon(mesh_poly) # to be used in covers_positions (does not include the islands)
        boundary_with_islands = Polygon(mesh_poly,holes = island_polys) # to be used as landmask (includes islands)

        if False: # check plot, and test in-polys checks
            from shapely.vectorized import contains
            import matplotlib.pyplot as plt
            plt.ion();plt.show()
            plt.plot(mesh_poly[:,0],mesh_poly[:,1])
            for isl in island_polys:plt.plot(isl[:,0],isl[:,1]) 
            xy=plt.ginput(10)
            isin = contains(boundary_with_islands, np.array(xy)[:,0], np.array(xy)[:,1])
            plt.plot(np.array(xy)[isin,0],np.array(xy)[isin,1],'ro')
            plt.plot(np.array(xy)[~isin,0],np.array(xy)[~isin,1],'go')
            import pdb;pdb.set_trace()

        # convert to prepared geometries
        boundary = prep(Polygon(boundary))
        boundary_with_islands = prep(Polygon(boundary_with_islands))
        return boundary,boundary_with_islands
        

    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None, block=False):

        """ The function extracts or generates the 'requested_variables' from the SCHISM cons file from Oceanum's Datamesh.
            For tidal <elev,u,v> predictions, we use the python package oceantide https://github.com/oceanum/oceantide/
            to generate gridded fields of elev,u,v from the tidal constituent grid.
            These fields are that are then passed to get_variables_interpolated() as if extracted from usual schism files
            for interpolation to particle positions by _get_variables_interpolated().

            An alternative would be to generate the elev, u,v at particle positions directly in this reader
            with a dedicated _get_variables_interpolated() which would overload the one below that is taken from
            reader_schism_native().

            To explore...we could also generate the gridded fields only for the extents of the particles clouds + some buffer
            see below

            For now the function will extract the entire slice of data of 'requested_variables' at given 'time'

            There is an option to extract only a subset of data around particles clouds to have less data but
            it means we need to recompute the KDtree of the subset nodes every time in ReaderBlockUnstruct.
            
            Speed gain to be tested ...
        """
        print(time)
        requested_variables, time, x, y, z, outside = \
            self.check_arguments(requested_variables, time, x, y, z)

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        variables = {'x': self.x, 'y': self.y, 'z': 0.*self.y,
                     'time': nearestTime}
        
        # generate the full variable slices of requested_variables at time indxTime

        # if tidal velocities are requested, we generate the flow field for that time
        if 'x_sea_water_velocity' in requested_variables:
            tide_pred = self.dataset.tide.predict(times=time)

        for par in requested_variables:
            var = self.dataset.variables[self.variable_mapping[par]]
            # there are only 4 <requested_variables> options 
            if par is 'sea_floor_depth_below_sea_level':
                if var.ndim == 1: # time-independent variable
                    data = var[:] # e.g. depth
                    logger.debug('reading constant data from unstructured reader %s' % (par))    
            elif par is 'x_sea_water_velocity':
                data = tide_pred.u
                logger.debug('reading 2D data from unstructured reader %s' % (par))
            elif par is 'y_sea_water_velocity':
                data = tide_pred.v
                logger.debug('reading 2D data from unstructured reader %s' % (par))
            elif par is 'sea_surface_height':
                data = tide_pred.h
                logger.debug('reading 2D data from unstructured reader %s' % (par))
            elif par is 'land_binary_mask':
                data = 0*var # allocate with 0 for now, check will be done using self.mesh_polygon
                logger.debug('reading 2D data from unstructured reader %s (set all to 0)' % (par))
            else:
                raise ValueError('Wrong dimension of %s: %i' %
                                    (self.variable_mapping[par], var.ndim))
            
            if False:
                import pdb;pdb.set_trace()
                import matplotlib.pyplot as plt 
                plt.ion()
                plt.show()
                plt.scatter(self.x,self.y,c=data)
                plt.plot(174.121503,-35.310339,'ko')

            variables[par] = data # save all data slice to dictionary with key 'par'
            # Store coordinates of returned points
            #  >> done in previous steps here, line 380 and in convert_3d_to_array()

            variables[par] = np.asarray(variables[par])

        self.use_subset = False # Functionnal now  - need to check results are consistent.
        if self.use_subset: # for testing subsetting data before computin KD-trees
            variables = self.clip_reader_data(variables_dict = variables, x_particle = x,y_particle = y, requested_variables = requested_variables) # clip reader data to particle cloud coverage 
            # update the 2D KDtree (will be used to initialize the ReaderBlockUnstruct)
            self.reader_KDtree = cKDTree(np.vstack((variables['x'],variables['y'])).T) 
            # the 3D KDtree in updated within ReaderBlockUnstruct()
        return variables 

    def clip_reader_data(self,variables_dict, x_particle,y_particle,requested_variables=None):
        # clip "variables_dict" to current particle cloud extents [x_particle,y_particle] (+ buffer)
        # 
        # # Find a subset of mesh nodes that include the particle 
        # cloud. The frame should not be made too small to make sure
        # particles remain inside the frame over the time 
        # that interpolator is used (i.e. depends on model timestep)
        # 
        # 
        #  returns updated "variables" dict and data array
        deg2meters = 1.1e5 # approx
        buffer = .1  # degrees around given positions ~10km
        buffer = .05  # degrees around given positions ~ 5km - reader seems quite sensitive to that buffer...
        if self.xmax <= 360 :
            pass # latlong reference system - keep same buffer value
        else:
            buffer = buffer * deg2meters# projected reference system in meters
        self.subset_frame = [x_particle.min() - buffer,
                             x_particle.max() + buffer, 
                             y_particle.min() - buffer,
                             y_particle.max() + buffer]
        logger.debug('Spatial frame used for schism reader %s (buffer = %s) ' % (str(self.subset_frame) , buffer))
        # find lon,lat of reader that are within the particle cloud
        self.id_frame = np.where((self.lon >= self.subset_frame[0]) &
                     (self.lon <= self.subset_frame[1]) &
                     (self.lat >= self.subset_frame[2]) &
                     (self.lat <= self.subset_frame[3]))[0]
        logger.debug('Using %s ' % (int(100*self.id_frame.shape[0]/self.lon.shape[0])) + '%' + ' of native nodes' )
        # print(variables_dict['x'].shape)
        # print( self.id_frame.max())

        if False:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.plot(x_particle,y_particle,'.')
            box = np.array([(self.subset_frame[0],self.subset_frame[2]),\
                            (self.subset_frame[1],self.subset_frame[2]),\
                            (self.subset_frame[1],self.subset_frame[3]),\
                            (self.subset_frame[0],self.subset_frame[3]),\
                            (self.subset_frame[0],self.subset_frame[2])])
            plt.plot(box[:,0],box[:,1])
            plt.title('frame used in clip_reader_data()')
            import pdb;pdb.set_trace()
            # plt.close()

        variables_dict['x'] = variables_dict['x'][self.id_frame]
        variables_dict['y'] = variables_dict['y'][self.id_frame]
        variables_dict['z'] = variables_dict['z'][self.id_frame]

        if 'x_3d' in variables_dict:
            ID= np.where((variables_dict['x_3d'] >= self.subset_frame[0]) &
                 (variables_dict['x_3d'] <= self.subset_frame[1]) &
                 (variables_dict['y_3d'] >= self.subset_frame[2]) &
                 (variables_dict['y_3d']<= self.subset_frame[3]))[0]  
            variables_dict['x_3d'] = variables_dict['x_3d'][ID]
            variables_dict['y_3d'] = variables_dict['y_3d'][ID]
            variables_dict['z_3d'] = variables_dict['z_3d'][ID]

        for par in requested_variables:
            if 'x_3d' not in variables_dict:
                # 2D data
                variables_dict[par] = variables_dict[par][self.id_frame]
            else: # can be either 2D or 3D data, in case requested_variables includes both 2D and 3D data
                if variables_dict[par].shape[0] == self.lon.shape[0] : # this "par" is 2D data
                    variables_dict[par] = variables_dict[par][self.id_frame]
                else:
                    # 3D data converted to array
                    variables_dict[par] = variables_dict[par][ID]

        return variables_dict
        
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
        if self.use_log_profile :
            self.apply_logarithmic_current_profile(env,z)
         
        # additional on-land checks using mesh_polygon (if present)
        if 'land_binary_mask' in env.keys() and self.use_mesh_polygon : #and hasattr(self,'shore_file'):
            logger.debug('Updating land_binary_mask using mesh polygon')
            lon_tmp,lat_tmp = self.xy2lonlat(reader_x,reader_y)
            # check if particles are within mesh polygon (if False, they are on land)
            in_mesh = shapely.vectorized.contains(self.mesh_polygon, lon_tmp, lat_tmp) 
            # update the 'land_binary_mask' accounting for the in-mesh checks (land_binary_mask==1 if particles are on land) 
            env['land_binary_mask'] = np.maximum(env['land_binary_mask'],np.invert(in_mesh).astype(float))

        # make sure dry points have zero velocities which is not always the case
        # we could also look at using depth and thresholds to flag other dry points ?
        if 'land_binary_mask' in env.keys() and \
            'x_sea_water_velocity' in env.keys() and \
            env['land_binary_mask'].astype('bool').any():

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
                 KDtree = None,
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

        # If we eventually use subset of nodes rather than full mesh, we'll need to re-compute the 2D KDtree
        # as well, instead of re-using the "full" one available from reader's init
        # logger.debug('Compute time-varying KDtree for 2D nearest-neighbor search') 
        # self.block_KDtree = cKDTree(np.vstack((self.x,self.y)).T)  # KDtree input to function = one computed during reader's __init__()

        if hasattr(self,'z_3d'):
            # we need to compute a new KDtree for that time step using vertical coordinates at that time step
            logger.debug('Compute time-varying KDtree for 3D nearest-neighbor search (i.e using ''zcor'') ')
            # clean arrays if needed, especially z_3d (get rid of nan's) - keep only non-nan
            # 
            # check for nan's 
            # if (self.z_3d != self.z_3d).any(): # not required
            #     self.x_3d = self.x_3d[np.where(self.z_3d == self.z_3d)]
            #     self.y_3d = self.y_3d[np.where(self.z_3d == self.z_3d)]
            #     self.z_3d = self.z_3d[np.where(self.z_3d == self.z_3d)]
            # check for infinite values
            if np.isinf(self.z_3d).any() :
                self.z_3d[np.where(np.isinf(self.z_3d))] = 15.0 #limit to +15.0m i.e. above msl

            self.block_KDtree_3d = cKDTree(np.vstack((self.x_3d,self.y_3d,self.z_3d)).T) 
            # do we need copy_data=True ..probably not since "data" [self.x_3d,self.y_3d,self.z_3d] 
            # will not change without the KDtree being recomputedplt

        # Mask any extremely large values, e.g. if missing netCDF _Fill_value
        filled_variables = set()
        for var in self.data_dict:
            if isinstance(self.data_dict[var], np.ma.core.MaskedArray):
                self.data_dict[var] = np.ma.masked_outside(
                    np.ma.masked_invalid(self.data_dict[var]), -1E+9, 1E+9)
                # Convert masked arrays to numpy arrays
                self.data_dict[var] = np.ma.filled(self.data_dict[var],
                                                   fill_value=np.nan)
            # Fill missing data towards seafloor if 3D
            if isinstance(self.data_dict[var], (list,)):
                logger.warning('Ensemble data currently not extrapolated towards seafloor')
            elif self.data_dict[var].ndim == 3:
                filled = fill_NaN_towards_seafloor(self.data_dict[var])
                if filled is True:
                    filled_variables.add(var)
                
        if len(filled_variables) > 0:
            logger.debug('Filled NaN-values toward seafloor for :'
                          + str(list(filled_variables)))
        
        # below probably not be relevant any longer
        if False:
            # Set 1D (vertical) and 2D (horizontal) interpolators
            try:
                self.Interpolator2DClass = \
                    horizontal_interpolation_methods[interpolation_horizontal]
            except Exception:
                raise NotImplementedError(
                    'Valid interpolation methods are: ' +
                    str(horizontal_interpolation_methods.keys()))

            try:
                self.Interpolator1DClass = \
                    vertical_interpolation_methods[interpolation_vertical]
            except Exception:
                raise NotImplementedError(
                    'Valid interpolation methods are: ' +
                    str(vertical_interpolation_methods.keys()))

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
            profiles_dict = {'z': self.z} # probably not valid...
        for varname, data in iteritems(self.data_dict):
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
                if data.shape[0] == self.x.shape[0] : # 2D data- full slice
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
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(self.x_3d[i[0]].data.tolist(),self.y_3d[i[0]].data.tolist(),self.z_3d[i[0]].data.tolist(),c='r', marker='o')
                        ax.scatter(x[0:1],y[0:1],z[0:1],c='g', marker='o')
                        plt.ion()
                        plt.show()
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
                # not functional yet...
                # need to lookup what actually is expected here
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
