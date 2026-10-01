import numpy as np
import pytest
import xarray as xr
from datetime import datetime

oceantide = pytest.importorskip('oceantide')

from opendrift.readers import reader_datamesh_schism_cons


@pytest.fixture
def cons_file(tmp_path):
    """Small synthetic unstructured constituent grid, in Datamesh format"""
    nx, ny = 10, 10
    lon, lat = np.meshgrid(np.linspace(5, 5.09, nx), np.linspace(60, 60.09, ny))
    lon, lat = lon.ravel(), lat.ravel()
    node = lambda ix, iy: iy * nx + ix
    boundary = [node(ix, 0) for ix in range(nx)] + \
               [node(nx - 1, iy) for iy in range(1, ny)] + \
               [node(ix, ny - 1) for ix in range(nx - 2, -1, -1)] + \
               [node(0, iy) for iy in range(ny - 2, 0, -1)]
    # one island, padded with NaN as in Datamesh files
    island = [[node(4, 4), node(5, 4), node(5, 5), node(4, 5), np.nan]]
    cons = ['M2  ', 'S2  ', 'K1  ']
    rng = np.random.default_rng(0)
    shape = (len(cons), nx * ny)
    ds = xr.Dataset({
        'lon': ('node', lon, {'standard_name': 'longitude'}),
        'lat': ('node', lat, {'standard_name': 'latitude'}),
        'dep': ('node', np.linspace(5, 50, nx * ny)),
        'boundary': ('bnode', np.float64(boundary)),
        'island': (('inum', 'inode'), np.array(island)),
        'cons': ('con', cons),
        **{f'{v}_{part}': (('con', 'node'), rng.normal(size=shape))
           for v in ['h', 'u', 'v'] for part in ['re', 'im']},
        })
    filename = tmp_path / 'tidalcons.zarr'
    ds.to_zarr(filename)
    return filename


def test_datamesh_schism_cons(cons_file):
    r = reader_datamesh_schism_cons.Reader(filename=cons_file)
    x = np.array([5.012, 5.033, 5.071, 5.045])  # last point is on island
    y = np.array([60.021, 60.058, 60.012, 60.045])
    time = datetime(2024, 6, 15, 7, 23)
    variables = ['x_sea_water_velocity', 'y_sea_water_velocity', 'sea_surface_height',
                 'sea_floor_depth_below_sea_level', 'land_binary_mask']
    env, _ = r._get_variables_interpolated_(variables, None, None, time, x, y, 0 * x)

    # Reference: tide predicted by oceantide on full mesh, then same interpolation
    ref = r.dataset.tide.predict(times=time).squeeze().load()
    dist, i = r.reader_KDtree.query(np.vstack((x, y)).T, 3)
    fac = 1. / dist
    for var, v in [('x_sea_water_velocity', 'u'), ('y_sea_water_velocity', 'v'),
                   ('sea_surface_height', 'h')]:
        expected = (fac * ref[v].values[i]).sum(-1) / fac.sum(-1)
        if var != 'sea_surface_height':
            expected[-1] = 0  # velocities are set to zero on land
        np.testing.assert_allclose(env[var], expected, atol=1e-12)
    np.testing.assert_allclose(env['sea_floor_depth_below_sea_level'],
                               (fac * r.dataset.dep.values[i]).sum(-1) / fac.sum(-1))
    np.testing.assert_array_equal(env['land_binary_mask'], [0, 0, 0, 1])
