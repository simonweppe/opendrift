##########################################################
# New functions for Lagrangian Coherent Structures
# Authors Simon Weppe - Calypso Science
##########################################################

    def calculate_green_cauchy_tensor(self,
                       reader=None,
                       delta=None,
                       domain=None,
                       time=None,
                       time_step=None,
                       duration=None,
                       z=0,
                       RLCS=True,
                       ALCS=True,
                       cartesian_epsg = None):
        """
        Calculate the Green-Cauchy tensor for Lagrangian Coherent Structures (LCS).

        Using the function calculate_ftle() as template and adding new code from 
        script here : https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L352
        by  Rodrigo Duran,Mireya Montano.

        This method computes the Green-Cauchy tensor based on Lagrangian particle trajectories
        derived from a opendrift object (that can have several readers). The tensor components 
        are calculated for repulsive (RLCS) (forward-time simulations) and 
        attractive (ALCS) LCS (backward-time simulations).

        Parameters:
        reader (str or pyproj.Proj): Data reader object or projection string (proj4). If None,
            the first available reader in the environment is used.
        delta (float): Spacing between seed points in the seeding domain (in meters).
        domain (list): [xmin, xmax, ymin, ymax] defining the spatial domain for seeding (same coords system as <reader>).
            If None, the domain is automatically determined based on the reader's extent.

        time (datetime or list): Start time or list of times for which LCS is computed.
        time_step (timedelta, or float): Time step used for numerical integration of trajectories.
        duration (timedelta, or float): Duration of particle advection for LCS computations.
        RLCS (bool): Whether to compute repulsive LCS (default: True).
        ALCS (bool): Whether to compute attractive LCS (default: True).
        cartesian_epsg (int): EPSG code of the Cartesian coordinate system to use for seeding
            when the reader's native coordinate system is lon/lat (WGS84).

        Returns:
        tuple: A tuple containing two items:
            - lcs (dict): Dictionary containing LCS components (RLCS and ALCS) and related metrics.
            - ds_lcs (xarray.Dataset): Dataset containing LCS data organized as xarray variables
            (RLCS, ALCS, C components, seeding coordinates, etc.).

        Notes:
        - This method utilizes Lagrangian trajectory integration to compute the Green-Cauchy tensor
        components based on forward and backward advection of particle positions.
        - The resulting LCS components are masked to handle invalid or NaN values.

        Examples:
        # Initialize LCS computation using default parameters
        lcs, ds_lcs = calculate_green_cauchy_tensor()

        # Compute LCS for a specific reader, time range, and domain
        lcs, ds_lcs = calculate_green_cauchy_tensor(reader=reader_current,
                                                    time=datetime(2024, 4, 8, 12),
                                                    duration=timedelta(minutes=15),
                                                    domain=(-1000, 1000, -1000, 1000),
                                                    cartesian_epsg=2193)
        """
        if reader is None:
            logger.info('No reader provided, using first available:')
            reader = list(self.env.readers.items())[0][1]
            logger.info(reader.name)
        if isinstance(reader, pyproj.Proj):
            proj = reader
        elif isinstance(reader, str):
            proj = pyproj.Proj(reader)
        # added simon, if proj4 is defined but self.proj was not instanciated
        elif hasattr(reader, 'proj4') :
            proj = pyproj.Proj(reader.proj4)
        else :
            proj = reader.proj

        # from opendrift.models.physics_methods import ftle

        if not isinstance(duration, timedelta):
            duration = timedelta(seconds=duration)

        # define seeding domain 
        if  not reader.proj.crs.to_epsg() == 4326 : #not np.logical_or(reader.proj.crs.to_epsg() == 4326) ,np.abs(reader.xmax)<360) :  
            # reader coordinates system is projected/cartesian, so we can use it 
            # to define the seeding domain
            if domain == None:
                xs = np.arange(reader.xmin, reader.xmax, delta)
                ys = np.arange(reader.ymin, reader.ymax, delta)
            else:
                xmin, xmax, ymin, ymax = domain
                xs = np.arange(xmin, xmax, delta)
                ys = np.arange(ymin, ymax, delta)
        else :
            # reader is natively in lon,lat, we need a user-defined coordinate system to use, 
            # to prepare the seeding frame 
            logger.info('reader native coordinate system is wgs4, applying user-defined <cartesian_epsg> to define seeding frame used in LCS computation')
            if cartesian_epsg is None :
                from pyproj import CRS
                logger.error('You need to specify a <cartesian_epsg> in calculate_green_cauchy_tensor() to define (cartesian) seeding grid')
                import pdb;pdb.set_trace()
            else:
                # define the cartesian projection
                proj = pyproj.Proj("EPSG:%s" % cartesian_epsg)
                if domain == None: # use entire reader extents as domain
                    # projected reader extents
                    xmin,ymin = proj(reader.xmin,reader.ymin)
                    xmax,ymax = proj(reader.xmax,reader.ymax)
                else: # use user-defined domain
                    xmin, xmax, ymin, ymax = domain
                    if xmax<=360: # if user defined the domain in lon,lat, we need to convert to cartesian
                        xmin,ymin = proj(xmin,ymin)
                        xmax,ymax = proj(xmax,ymax)
                        logger.info('converting user-defined LCS domain from WGS84 (%s) to cartesian (%s)' % (domain,[xmin,xmax,ymin,ymax]))
                        logger.info('Note domain must be defined as [Xmin,Xmax,Ymin,Ymax]')
                xs = np.arange(xmin, xmax, delta)
                ys = np.arange(ymin, ymax, delta)  
        logger.debug('seeding frame for LCS computation (cartesian) [xmin=%s,xmax=%s,ymin=%s,ymax=%s]' % (xs[0],xs[-1],ys[0],ys[-1]))
        # final seeding frame
        X, Y = np.meshgrid(xs, ys)
        lons, lats = proj(X, Y, inverse=True)

        if time is None:
            time = reader.start_time
        if not isinstance(time, list):
            time = [time]
        # dictionary to hold LCS calculation
        lcs = {'time': time, 'lon': lons, 'lat': lats}
        # LCS and C variables for repulsive LCS
        lcs['RLCS'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['R_lda2'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['R_C11'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['R_C12'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['R_C22'] = np.zeros((len(time), len(ys), len(xs)))
        # LCS and C variables for attractive LCS
        lcs['ALCS'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['A_lda2'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['A_C11'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['A_C12'] = np.zeros((len(time), len(ys), len(xs)))
        lcs['A_C22'] = np.zeros((len(time), len(ys), len(xs)))

        T = np.abs(duration.total_seconds())

        for i, t in enumerate(time):
            logger.info('Calculating LCS for ' + str(t))
            # Forwards
            if RLCS is True:
                o = self.clone() # full reset of opendrift object
                o.seed_elements(lons.ravel(), lats.ravel(), time=t, z=z)
                o.run(duration=duration, time_step=time_step)
                # f_x1, f_y1 = proj(o.history['lon'].T[-1].reshape(X.shape),
                #                   o.history['lat'].T[-1].reshape(X.shape))
                f_x1, f_y1 = proj(o.result['lon'].data.T[-1].reshape(X.shape),
                                  o.result['lat'].data.T[-1].reshape(X.shape))
                lcs['R_C11'][i, :, :], \
                lcs['R_C12'][i, :, :], \
                lcs['R_C22'][i, :, :], \
                lcs['R_lda2'][i, :, :], \
                lcs['RLCS'][i, :, :] =  GC_tensor(f_x1 - X, f_y1 - Y, delta, T)
                # lcs['RLCS'][i, :, :] = ftle(f_x1 - X, f_y1 - Y, delta, T)

            # Backwards
            if ALCS is True:
                o = self.clone() # full reset of opendrift object
                o.seed_elements(lons.ravel(),
                                lats.ravel(),
                                time=t + duration,
                                z=z)
                o.run(duration=duration, time_step=-time_step)
                # when run in backwards mode, opendrift inverts the order of lon/lat items in the array
                # so we flip the array  to be consistent with initial positions, and compute correct displacements
                # See similar comment : https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L335

                # b_x1, b_y1 = proj(o.history['lon'].T[-1][::-1].reshape(X.shape),
                #                   o.history['lat'].T[-1][::-1].reshape(X.shape))                
                b_x1, b_y1 = proj(o.result['lon'].data.T[-1][::-1].reshape(X.shape),
                                  o.result['lat'].data.T[-1][::-1].reshape(X.shape))  

                lcs['A_C11'][i, :, :], \
                lcs['A_C12'][i, :, :], \
                lcs['A_C22'][i, :, :], \
                lcs['A_lda2'][i, :, :], \
                lcs['ALCS'][i, :, :] =  GC_tensor(b_x1 - X, b_y1 - Y, delta, T)
                # lcs['ALCS'][i, :, :] = ftle(b_x1 - X, b_y1 - Y, delta, T)

        # it seems we can mask values where ALCS/RCLS == 1.0, and C**== 0, and lda2==0.5
        lcs['ALCS'][lcs['ALCS']==1.0] = np.nan
        lcs['A_C11'][lcs['A_C11']==0.0] = np.nan
        lcs['A_C12'][lcs['A_C12']==0.0] = np.nan
        lcs['A_C22'][lcs['A_C22']==0.0] = np.nan
        lcs['A_lda2'][lcs['A_lda2']==0.5] = np.nan

        lcs['RLCS'][lcs['RLCS']==1.0] = np.nan
        lcs['R_C11'][lcs['R_C11']==0.0] = np.nan
        lcs['R_C12'][lcs['R_C12']==0.0] = np.nan
        lcs['R_C22'][lcs['R_C22']==0.0] = np.nan
        lcs['R_lda2'][lcs['R_lda2']==0.5] = np.nan

        # mask arrays      
        for var in lcs.keys():
            if var not in ['time', 'lon', 'lat']:
                lcs[var] = np.ma.masked_invalid(lcs[var])

        # Mask landpoints
        # There must be a better way to do this on xarray object
        # using advanced indexing e.g.
        # https://gist.github.com/robbibt/17bef639a26acf3c86091ac64b8c4bb6
        if True:
            logger.debug('Masking land points for LCS ')
            self.env.finalize()
            landmask = self.env.get_environment(['land_binary_mask'],
                                                lon=lons.ravel(), 
                                                lat=lats.ravel(),
                                                z=0.0*lons.ravel(),
                                                time=time[0],
                                                profiles=None)
            landmask = np.array(landmask[0].tolist()).squeeze().reshape(lons.shape)
            landmask[np.isnan(landmask)]=1 # in case some nans are returned
            # tile along time dimension
            landmask_all = np.tile(landmask,(len(time),1,1))
            lcs['RLCS'][landmask_all==1] = np.nan
            lcs['ALCS'][landmask_all==1] = np.nan

        # Flipping ALCS left-right. Not sure why this is needed
        # It's not needed for RLCS
        # 
        # >> not needed anymore now that we use re-ordered the (lon,lat) earlier for ALCS
        # 
        # lcs['ALCS'] = lcs['ALCS'][:, ::-1, ::-1]
        # lcs['A_C11'] = lcs['A_C11'][:, ::-1, ::-1]
        # lcs['A_C12'] = lcs['A_C12'][:, ::-1, ::-1]
        # lcs['A_C22'] = lcs['A_C22'][:, ::-1, ::-1]
        # lcs['A_lda2'] = lcs['A_lda2'][:, ::-1, ::-1]
        
        # Convert LCS data to xarray dataset
        import xarray as xr
        data_dict = {  'ALCS': (('time', 'lat', 'lon'), lcs['ALCS'].data,{'units': '-', 'description': 'FTLE attractive LCS'} ),
                       'A_C11': (('time', 'lat', 'lon'), lcs['A_C11'].data,{'units': '-', 'description': 'C11 constant attractive'} ),
                       'A_C12': (('time', 'lat', 'lon'), lcs['A_C12'].data,{'units': '-', 'description': 'C12 constant attractive'} ),
                       'A_C22': (('time', 'lat', 'lon'), lcs['A_C22'].data,{'units': '-', 'description': 'C22 constant attractive'} ),
                       'A_lda2' : (('time', 'lat', 'lon'), lcs['A_lda2'].data,{'units': '-', 'description': 'lda2 attractive LCS'} ),
              
                       'RLCS': (('time', 'lat', 'lon'), lcs['RLCS'].data,{'units': '-', 'description': 'FTLE repulsive LCS'}),
                       'R_C11': (('time', 'lat', 'lon'), lcs['R_C11'].data,{'units': '-', 'description': 'C11 constant repulsive'} ),
                       'R_C12': (('time', 'lat', 'lon'), lcs['R_C12'].data,{'units': '-', 'description': 'C12 constant repulsive'} ),
                       'R_C22': (('time', 'lat', 'lon'), lcs['R_C22'].data,{'units': '-', 'description': 'C22 constant repulsive'} ),
                       'R_lda2' : (('time', 'lat', 'lon'), lcs['R_lda2'].data,{'units': '-', 'description': 'lda2 repulsive LCS'} ),

                       'X' : (( 'lat', 'lon'), X,{'units': 'm', 'description': 'cartesian seeding x-coord'} ),
                       'Y' : (( 'lat', 'lon'), Y,{'units': 'm', 'description': 'cartesian seeding y-coord'} ),
                       }  
        ds_lcs = xr.Dataset(data_vars=data_dict, 
                        coords={'lon2D': (('lat', 'lon'), lcs['lon']), 'lat2D': (('lat', 'lon'), lcs['lat']), 'time': lcs['time']})
        return lcs, ds_lcs

    def calculate_LD(self,
                        reader=None,
                        delta=None,
                        domain=None,
                        time=None,
                        time_step=None,
                        duration=None,
                        z=0,
                        cartesian_epsg = None):
            """
            Calculate <Lagrangian Descriptors> to inform on Lagrangian Coherent Structures (LCS).

            Mendoza, C., & Mancho, A. M. (2010). Hidden geometry of ocean flows. Physical review letters, 105(3), 038501.

            The function provides the "M function" which is is a particular case of a Lagrangian Descriptor that is computed by 
            integrating the particle trajectory distance over its lifetime from t0-integration_time to t0+integration_time

            A skeleton of the stable and unstable manifolds of hyperbolic trajectories in time dependent flows 
            can be constructed with the technique that is referred to as Lagrangian descriptors, based on the function M.
            Relative to FTLE this allows capturing both hyperbolic and non-hyperbolic regions (repulsive/attractive)
            
            (García-Garrido, V. J., Ramos, A., Mancho, A. M., Coca, J., & Wiggins, S. (2016). A dynamical systems perspective 
            for a real-time response to a marine oil spill. Marine pollution bulletin, 112(1-2), 201-210.)
            
            There are other Lagrangian Descriptors. See online book :
            https://champsproject.github.io/lagrangian_descriptors/docs/authors.html
            
            with codes:
            https://champsproject.github.io/lagrangian_descriptors/content/examples.html

            Parameters:
            reader (str or pyproj.Proj): Data reader object or projection string (proj4). If None,
                the first available reader in the environment is used.
            delta (float): Spacing between seed points in the seeding domain (in meters).
            domain (list): [xmin, xmax, ymin, ymax] defining the spatial domain for seeding (same coords system as <reader>).
                If None, the domain is automatically determined based on the reader's extent.

            time (datetime or list): Start time or list of times for which LCS is computed.
            time_step (timedelta, or float): Time step used for numerical integration of trajectories.
            duration (timedelta, or float): Duration of particle advection for LCS computations.
            cartesian_epsg (int): EPSG code of the Cartesian coordinate system to use for seeding
                when the reader's native coordinate system is lon/lat (WGS84).

            Returns:
            tuple: A tuple containing two items:
                - LD (dict): Dictionary containing the Lagrangian Descriptors (LD)
                - ds_LD (xarray.Dataset): Dataset containing LD data with M function

            Examples:
            # Initialize LCS computation using default parameters
            lcs, ds_lcs = calculate_LD()

            # Compute LCS for a specific reader, time range, and domain
            lcs, ds_lcs = calculate_LD( reader=reader_current,
                                        time=datetime(2024, 4, 8, 12),
                                        duration=timedelta(minutes=15),
                                        domain=(-1000, 1000, -1000, 1000),
                                        cartesian_epsg=2193)
            """

            if reader is None:
                logger.info('No reader provided, using first available:')
                reader = list(self.env.readers.items())[0][1]
                logger.info(reader.name)
            if isinstance(reader, pyproj.Proj):
                proj = reader
            elif isinstance(reader, str):
                proj = pyproj.Proj(reader)
            # added simon, if proj4 is defined but self.proj was not instanciated
            elif hasattr(reader, 'proj4') :
                proj = pyproj.Proj(reader.proj4)
            else :
                proj = reader.proj

            # from opendrift.models.physics_methods import ftle

            if not isinstance(duration, timedelta):
                duration = timedelta(seconds=duration)

            # define seeding domain 
            if  not reader.proj.crs.to_epsg() == 4326 : #not np.logical_or(reader.proj.crs.to_epsg() == 4326) ,np.abs(reader.xmax)<360) :  
                # reader coordinates system is projected/cartesian, so we can use it 
                # to define the seeding domain
                if domain == None:
                    xs = np.arange(reader.xmin, reader.xmax, delta)
                    ys = np.arange(reader.ymin, reader.ymax, delta)
                else:
                    xmin, xmax, ymin, ymax = domain
                    xs = np.arange(xmin, xmax, delta)
                    ys = np.arange(ymin, ymax, delta)
            else :
                # reader is natively in lon,lat, we need a user-defined coordinate system to use, 
                # to prepare the seeding frame 
                logger.info('reader native coordinate system is wgs4, applying user-defined <cartesian_epsg> to define seeding frame used in LCS computation')
                if cartesian_epsg is None :
                    from pyproj import CRS
                    logger.error('You need to specify a <cartesian_epsg> in calculate_green_cauchy_tensor() to define (cartesian) seeding grid')
                    import pdb;pdb.set_trace()
                else:
                    # define the cartesian projection
                    proj = pyproj.Proj("EPSG:%s" % cartesian_epsg)
                    if domain == None: # use entire reader extents as domain
                        # projected reader extents
                        xmin,ymin = proj(reader.xmin,reader.ymin)
                        xmax,ymax = proj(reader.xmax,reader.ymax)
                    else: # use user-defined domain
                        xmin, xmax, ymin, ymax = domain
                        if xmax<=360: # if user defined the domain in lon,lat, we need to convert to cartesian
                            xmin,ymin = proj(xmin,ymin)
                            xmax,ymax = proj(xmax,ymax)
                            logger.info('converting user-defined LCS domain from WGS84 (%s) to cartesian (%s)' % (domain,[xmin,xmax,ymin,ymax]))
                            logger.info('Note domain must be defined as [Xmin,Xmax,Ymin,Ymax]')
                    xs = np.arange(xmin, xmax, delta)
                    ys = np.arange(ymin, ymax, delta)  
            logger.debug('seeding frame for LCS computation (cartesian) [xmin=%s,xmax=%s,ymin=%s,ymax=%s]' % (xs[0],xs[-1],ys[0],ys[-1]))
            # final seeding frame
            X, Y = np.meshgrid(xs, ys)
            lons, lats = proj(X, Y, inverse=True)

            if time is None:
                time = reader.start_time
            if not isinstance(time, list):
                time = [time]
            # dictionary to hold M function
            LD = {'time': time, 'lon': lons, 'lat': lats}
            # LCS and C variables for repulsive LCS
            LD['M'] = np.zeros((len(time), len(ys), len(xs)))
            T = np.abs(duration.total_seconds())

            for i, t in enumerate(time):
                logger.info('Calculating LD for ' + str(t))
                # simulations must be run from t0-integration_time to t0+integration_time
                # ie. total run time = 2 * integration_time
                seed_time = t - duration
                o = self.clone() # full reset of opendrift object
                o.seed_elements(lons.ravel(), lats.ravel(), time = seed_time, z=z)
                o.run(duration= 2 * duration, time_step=time_step)
                # now compute arc_length along each individual particles
                total_length, distances, speeds = o.get_trajectory_lengths()
                M = total_length.reshape(X.shape)
                # M.mask=[M==0] # mask the array with land
                LD['M'][i, :, :] = M 
                
            # mask arrays      
            for var in LD.keys():
                if var not in ['time', 'lon', 'lat']:
                    LD[var] = np.ma.masked_invalid(LD[var])

            # Mask landpoints
            # There must be a better way to do this on xarray object
            # using advanced indexing e.g.
            # https://gist.github.com/robbibt/17bef639a26acf3c86091ac64b8c4bb6
            if False:
                # is it needed ? 
                logger.debug('Masking land points for LCS ')
                self.env.finalize()
                landmask = self.env.get_environment(['land_binary_mask'],
                                                    lon=lons.ravel(), 
                                                    lat=lats.ravel(),
                                                    z=0.0*lons.ravel(),
                                                    time=time[0],
                                                    profiles=None)
                landmask = np.array(landmask[0].tolist()).squeeze().reshape(lons.shape)
                landmask[np.isnan(landmask)]=1 # in case some nans are returned
                # tile along time dimension
                landmask_all = np.tile(landmask,(len(time),1,1))
                LD['M'][landmask_all==1] = np.nan
            
            # Convert LCS data to xarray dataset
            import xarray as xr
            data_dict = {  'M': (('time', 'lat', 'lon'), LD['M'].data,{'units': '-', 'description': 'Lagrangian Descriptor - M function'}),
                        'X' : (( 'lat', 'lon'), X,{'units': 'm', 'description': 'cartesian seeding x-coord'} ),
                        'Y' : (( 'lat', 'lon'), Y,{'units': 'm', 'description': 'cartesian seeding y-coord'} ),
                        }  
            ds_LD = xr.Dataset(data_vars=data_dict, 
                            coords={'lon2D': (('lat', 'lon'), LD['lon']), 'lat2D': (('lat', 'lon'), LD['lat']), 'time': LD['time']})
            return LD, ds_LD

def GC_tensor(X, Y, delta, duration):

    """Calculate Green Cauchy Tensors
    
        Adapted code from cLCS : https://github.com/MireyaMMO/cLCS/blob/main/cLCS/mean_C.py#L352

        Input : 
        X,Y : particle displacement at end of simulation (i.e. X = Xfinal-X0, Y=Yfinal-Y0) 
        delta : spatial seeding step (same on X and Y directions)
        duration : duration of simulation in seconds

        Outputs: 
        C11, C12, C22, lda2, ftle
        
        ftle should be consistent with results from ftle() in physics_methods.py
    """
    
    # self.logger.info("--- Calculating the Cauchy-Green Tensor")
    logger.debug("--- Calculating Cauchy-Green Tensor")
    #  original code
    # dxdy, dxdx = np.gradient(end_x, self.dy0, self.dx0)
    # dydy, dydx = np.gradient(end_y, self.dy0, self.dx0)
    # dxdx0 = np.reshape(dxdx, (self.Ny0, self.Nx0))
    # dxdy0 = np.reshape(dxdy, (self.Ny0, self.Nx0))
    # dydx0 = np.reshape(dydx, (self.Ny0, self.Nx0))
    # dydy0 = np.reshape(dydy, (self.Ny0, self.Nx0))

    # here dx0,dy0 equals to delta. The input (X,Y) are the total particle displacement at end of simulation,
    # and are already in matrix form, i.e. no need to reshape
    
    dxdy, dxdx =  np.gradient(X, delta, delta)
    dydy, dydx =  np.gradient(Y, delta, delta)
    
    dxdx0 = dxdx
    dxdy0 = dxdy
    dydx0 = dydx
    dydy0 = dydy
    
    C11 = (dxdx0**2) + (dydx0**2)
    C12 = (dxdx0 * dxdy0) + (dydx0 * dydy0)
    C22 = (dxdy0**2) + (dydy0**2)
    detC = (C11 * C22) - (C12**2)
    trC = C11 + C22
    lda2 = np.real(0.5 * trC + np.sqrt(0.25 * trC**2 - detC))
    ftle = np.log(lda2) / (2 * np.abs(duration))

    return C11, C12, C22, lda2, ftle