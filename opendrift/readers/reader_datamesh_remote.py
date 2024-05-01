import os
import datetime
import pyproj
import pandas as pd
import numpy as np
from oceanum.datamesh import Connector
from opendrift.readers.basereader import BaseReader, ContinuousReader, StructuredReader


datamesh = Connector()


class DataException(Exception):
    pass


class DatameshReader(BaseReader, StructuredReader):
    name = "datamesh"

    def __init__(self, datasource_id, mapping={}):
        self.proj4 = "+proj=lonlat +ellps=WGS84"  # Only working for WGS84 grids
        self.proj = pyproj.Proj(self.proj4)
        self.mapping = mapping
        try:
            self.dset = datamesh.load_datasource(datasource_id)
            self._vars = list(self.dset.variables.keys())
            self.variables = [v for v in mapping if mapping[v] in self._vars]
            self.latax, self.lonax = self._get_grid_axes()
            self.lon = self.dset[self.lonax].values
            self.lat = self.dset[self.latax].values
            self.time = self.dset["time"].values
            self.xmin = self.lon.min()
            self.xmax = self.lon.max()
            self.ymin = self.lat.min()
            self.ymax = self.lat.max()
            self.delta_x = self.lon[1] - self.lon[0]
            self.delta_y = self.lat[1] - self.lat[0]
            self.start_time = (
                pd.to_datetime(self.time[0]).tz_localize("UTC").to_pydatetime()
            )
            self.end_time = (
                pd.to_datetime(self.time[-1]).tz_localize("UTC").to_pydatetime()
            )
            self.time_step = (
                pd.to_datetime(self.time[1]) - pd.to_datetime(self.time[0])
            ).to_pytimedelta()

        except Exception as e:
            raise DataException(e)

        # Run constructor of parent Reader class
        super(DatameshReader, self).__init__()
        # Do this again to reset UTC
        self.start_time = (
            pd.to_datetime(self.time[0]).tz_localize("UTC").to_pydatetime()
        )
        self.end_time = pd.to_datetime(self.time[-1]).tz_localize("UTC").to_pydatetime()

    def _get_grid_axes(self):
        da_tmp = self.dset.get(self.mapping[self.variables[0]])
        lonax = da_tmp.dims[-1]
        latax = da_tmp.dims[-2]
        return latax, lonax

    def get_variables(self, requested_variables, time=None, x=None, y=None, z=None):

        requested_variables, time, x, y, z, outside = self.check_arguments(
            requested_variables, time, x, y, z
        )

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = self.nearest_time(time)

        variables = {}
        delta = self.buffer * self.delta_x
        lonmin = np.maximum(x.min() - delta, self.xmin)
        lonmax = np.minimum(x.max() + delta, self.xmax)
        latmin = np.maximum(y.min() - delta, self.ymin)
        latmax = np.minimum(y.max() + delta, self.ymax)

        if self.delta_y > 0:
            latslice = slice(latmin, latmax)
        else:
            latslice = slice(latmax, latmin)
        lonslice = slice(lonmin, lonmax)

        for var in requested_variables:
            subset = self.dset.get(self.mapping[var])
            subset = subset.isel(time=indxTime)
            variables[var] = subset.sel(
                {self.latax: latslice, self.lonax: lonslice}
            ).values
        variables["x"] = self.dset[self.lonax].sel({self.lonax: lonslice}).values
        variables["y"] = self.dset[self.latax].sel({self.latax: latslice}).values
        variables["z"] = None
        variables["time"] = nearestTime

        return variables
