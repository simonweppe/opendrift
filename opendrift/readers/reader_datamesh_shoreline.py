import os
import pygeos
import geopandas
from collections import OrderedDict
from sqlalchemy import create_engine
from opendrift.readers.basereader import BaseReader, ContinuousReader, StructuredReader

from oceanum.datamesh import Connector

datamesh = Connector()


class ShorelineException(Exception):
    pass


# This reader utilises the shoreline database
class ShorelineReader(BaseReader, ContinuousReader):
    name = "shoreline"
    variables = ["land_binary_mask"]
    proj4 = None
    crs = None
    skippoly = False
    datasource_select = OrderedDict(
        {
            1.0: "osm-land-polygons",
            5.0: "gshhs_f_l1",
            20.0: "gshhs_h_l1",
            50.0: "gshhs_i_l1",
            180.0: "gshhs_c_l1",
            1000.0: "gshhs_l_l1",
        }
    )

    def __init__(self, extent=None, skippoly=False):
        self.proj4 = "+proj=lonlat +ellps=WGS84"
        self.skippoly = skippoly

        super(ShorelineReader, self).__init__()
        self.z = None
        if extent is not None:
            self.xmin, self.ymin, self.xmax, self.ymax = extent
        else:
            self.xmin, self.ymin = -180, -90
            self.xmax, self.ymax = 180, 90

        domain_size = 1.0 * max(self.xmax - self.xmin, self.ymax - self.ymin)
        for res in self.datasource_select:
            if domain_size <= res:
                datasource = self.datasource_select[res]
                break

        query = {
            "datasource": datasource,
            "geofilter": {
                "type": "bbox",
                "geom": [self.xmin, self.ymin, self.xmax, self.ymax],
            },
        }

        self.shorelines = datamesh.query(query)

    def __on_land__(self, x, y):
        points = geopandas.GeoDataFrame({"geometry": pygeos.creation.points(x, y)})
        test = geopandas.sjoin(self.shorelines, points, how="right")
        return test.index_left >= 0

    def get_variables(self, requestedVariables, time=None, x=None, y=None, z=None):
        self.check_arguments(requestedVariables, time, x, y, z)
        return {"land_binary_mask": self.__on_land__(x, y)}
