import numpy as np
import pytest
import xarray as xr
from datetime import datetime

oceantide = pytest.importorskip('oceantide')

from opendrift.readers import reader_datamesh_regular_cons


@pytest.fixture
def cons_file(tmp_path):
    """Small synthetic regular constituent grid, in Datamesh format, in Quiberon Bay"""
    lon = np.linspace(-3.30, -3.00, 31)
    lat = np.linspace(47.30, 47.55, 26)
    cons = ['M2', 'S2', 'K1']
    rng = np.random.default_rng(0)
    shape = (len(cons), len(lat), len(lon))
    ds = xr.Dataset({
        f'{v}_{part}': (('con', 'lat', 'lon'), rng.normal(size=shape))
        for v in ['h', 'u', 'v'] for part in ['re', 'im']},
        coords={'con': cons, 'lat': lat, 'lon': lon})
    for v in ['h', 'u', 'v']:
        ds[f'{v}_re'][:, 10, 10] = np.nan  # land cell
    filename = tmp_path / 'tidalcons.zarr'
    ds.to_zarr(filename)
    return filename


def test_datamesh_regular_cons(cons_file):
    r = reader_datamesh_regular_cons.Reader(filename=cons_file)
    r.cons_block_margin = 2
    variables = ['x_sea_water_velocity', 'y_sea_water_velocity', 'sea_surface_height']
    time = datetime(2024, 6, 15, 7, 23)
    # last points are next to land cell (lon -3.2, lat 47.4), and outside grid
    x = np.array([-3.157, -3.121, -3.172, -3.195, -3.40])
    y = np.array([47.434, 47.451, 47.412, 47.405, 47.43])
    # second set of positions outside of window loaded for first set
    for x, y in [(x, y), (x + 0.1, y + 0.05)]:
        env, _ = r._get_variables_interpolated_(variables, None, None, time, x, y, 0 * x)
        assert r.cons_block['ix'][0] <= np.searchsorted(r.Dataset.lon.values, x[:-1].min())

        # Reference: constituents interpolated with xarray, then tide predicted by oceantide
        ref = r.Dataset.interp(lon=xr.DataArray(x, dims='z'),
                               lat=xr.DataArray(y, dims='z')).tide.predict(times=time)
        for var, v in [('x_sea_water_velocity', 'u'), ('y_sea_water_velocity', 'v'),
                       ('sea_surface_height', 'h')]:
            np.testing.assert_allclose(np.ma.filled(env[var], np.nan),
                                       ref[v].values.ravel(), atol=1e-12)
        if x[-1] < r.Dataset.lon.values.min():  # outside grid
            assert np.isnan(np.ma.filled(env['x_sea_water_velocity'], np.nan)[-1])
