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

# [name_used_in_schism : equivalent_CF_name]
schism_mapping = {
    'u': 'x_sea_water_velocity', 
    'v': 'y_sea_water_velocity',
    'dep': 'sea_floor_depth_below_sea_level',
    'h' : 'sea_surface_height',
    'dep': 'land_binary_mask',
    }
# do an inverted version
standard_name_mapping_datamesh_invert = {v: k for k, v in schism_mapping.items()}

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
        self.return_block = True

        try:
            # Open file, check that everything is ok
            logger.info('Opening dataset: ' + filestr)
            if ('.nc' not in filestr) :
                logger.info('Opening files with open_zarr')
                self.dataset = xr.open_zarr(filestr)
            else:
                logger.info('Opening netcdf file with open_dataset')
                import pdb;pdb.set_trace() # not tested yet
                self.dataset = xr.open_dataset(filestr,chunks={'time': 1})
            # need to edit the cons name for correct use in oceantide later on
            self.dataset['con']=[x.strip().upper() for x in self.dataset['cons'].values]
            self.check_tidal_amp_format() # check tidal amp/pha format and convert to complex numbers if necessary

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

    def check_tidal_amp_format(self):
        # check in which format the tidal consistuents' amplitude and phases are stored
        if 'h_im' in self.dataset.variables :
            # we need to convert into complex amplitudes
            #  as in oceantide :
            # https://github.com/oceanum/oceantide/blob/master/oceantide/input/oceantide.py
            for v in ["h", "u", "v"]:
                self.dataset[v] = self.dataset[f"{v}_re"] + 1j * self.dataset[f"{v}_im"] # > make sure to use +j here.
        elif 'e_pha' in self.dataset.variables :
            print('check tide cons file format - not tested yet')
            import pdb;pdb.set_trace()


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
       
    

    def nearest_time(self, time):
        """ overloads version from variables.py
        
        Original function : Return nearest times before and after the requested time.

        Here : we return the input time as nearest time as tide can be generated for any time
        Note this will not lead to time interpolation in _get_variables_interpolated_()
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
                      x=None, y=None, z=None, block=False):
        return None


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

            We generate tidal signal first, then interpolate to particles positions 
            using saved KDtree
            (instead of interpolating from gridded fields created in get_variables() )

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

        # tide_pred = self.dataset.interp(lon=lon_id, lat=lat_id).tide.predict(times=time).load()
        # tide_pred = self.dataset.tide.predict(times=time).squeeze() # we squeeze to get rid of dimension time

        # interp options
        nb_closest_nodes = 3
        DMIN=1.e-10
        dist,i=self.reader_KDtree.query(np.vstack((reader_x,reader_y)).T,nb_closest_nodes, workers=-1) #quick nearest-neighbor lookup
        # dist = distance to nodes / i = index of nodes
        dist[dist<DMIN]=DMIN
        fac=(1./dist)
        
        # # standard_name_mapping_datamesh_invert[vv]
        # import matplotlib.pyplot as plt;plt.ion();plt.show()
        # plt.plot(self.dataset.lon,self.dataset.lat,'k.')
        # plt.plot(reader_x,reader_y,'r.')
        # plt.plot(self.dataset.lon.isel(node=i[0]),self.dataset.lat.isel(node=i[0]),'g.')
        # plt.plot(self.dataset.lon.isel(node=i[1]),self.dataset.lat.isel(node=i[1]),'g.')
        # plt.plot(self.dataset.lon.isel(node=i[2]),self.dataset.lat.isel(node=i[2]),'g.')

        env = {}
        # the <env> variable to return is a dict such as
        # env =  {'sea_floor_depth_below_sea_level' : np.array(), ...}

        for vv in variables:
            if vv in ['x_sea_water_velocity','y_sea_water_velocity','sea_surface_height']:
                tide_pred = self.dataset.tide.predict(times=time).squeeze() # we squeeze to get rid of dimension time
                # interpolate to surrounding points
                data = tide_pred[standard_name_mapping_datamesh_invert[vv]].isel(node=i.ravel()).values.reshape(*i.shape)
                # linear interp to particles
                data_interpolated = (fac*data).sum(-1)/fac.sum(-1)
                env[vv] = np.ma.masked_invalid(data_interpolated)
            elif vv in ['sea_floor_depth_below_sea_level'] : # only depth can be requested as static variables 
                # interpolate to surrounding points
                data = self.dataset['dep'].isel(node=i.ravel()).values.reshape(*i.shape)
                # linear interp to particles
                data_interpolated = (fac*data).sum(-1)/fac.sum(-1)
                env[vv] = np.ma.masked_invalid(data_interpolated)
            elif vv in ['land_binary_mask']: # we enforce it here as the  self.activate_environment_mapping('land_binary_mask_from_ocean_depth') doesnt seem to work
                # interpolate to surrounding points
                data = self.dataset['dep'].isel(node=i.ravel()).values.reshape(*i.shape)
                # linear interp to particles
                data_interpolated = (fac*data).sum(-1)/fac.sum(-1)
                dep = np.ma.masked_invalid(data_interpolated)
                env[vv] = np.float32(dep <= 0) 
            else:
                # should not happen for now
                import pdb;pdb.set_trace()
        ######################################################################
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
        
        # not supporting profiles for now - set to None
        env_profiles = None

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
        ind_covered = np.where(self.covers_positions(x, y))[0] # this will call covers_positions() from unstructured.py
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
 

