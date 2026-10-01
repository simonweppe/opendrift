import numpy as np
import pandas as pd
import pytest
import xarray as xr

from opendrift.readers import reader_schism_native

proj4_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'


@pytest.fixture
def schism_file(tmp_path):
    """Small synthetic native SCHISM 3D output, on a triangular mesh with an island"""
    nx, ny, nlev, nt = 30, 25, 5, 3
    dx = 200.
    xx, yy = np.meshgrid(1700000 + dx * np.arange(nx), 5450000 + dx * np.arange(ny))
    x, y = xx.ravel(), yy.ravel()
    node = lambda i, j: j * nx + i
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if 10 <= i < 14 and 8 <= j < 12:
                continue  # island
            faces.append([node(i, j), node(i + 1, j), node(i + 1, j + 1)])
            faces.append([node(i, j), node(i + 1, j + 1), node(i, j + 1)])
    # faces next to island first: with this order, the island is the first polygon
    # found from mesh boundary edges, which must not be used as mesh outline
    faces = np.array(faces)
    centre = np.hypot(x[faces].mean(axis=1) - x[node(12, 10)], y[faces].mean(axis=1) - y[node(12, 10)])
    faces = faces[np.argsort(centre, kind='stable')].astype(float) + 1  # 1-based in SCHISM files
    faces = np.hstack((faces, np.full((len(faces), 1), np.nan)))
    rng = np.random.default_rng(0)
    depth = 20 + 30 * rng.random(len(x))
    elev = 0.5 * rng.normal(size=(nt, len(x)))
    sigma = np.linspace(-1, 0, nlev)
    zcor = elev[:, :, None] + sigma[None, None, :] * (depth[None, :, None] + elev[:, :, None])
    zcor[:, ::7, 0] = np.nan  # level below seabed for some nodes
    hvel = rng.normal(size=(nt, len(x), nlev, 2))
    ds = xr.Dataset({
        'SCHISM_hgrid_node_x': ('nSCHISM_hgrid_node', x, {'standard_name': 'projection_x_coordinate'}),
        'SCHISM_hgrid_node_y': ('nSCHISM_hgrid_node', y, {'standard_name': 'projection_y_coordinate'}),
        'SCHISM_hgrid_face_nodes': (('nSCHISM_hgrid_face', 'nMaxSCHISM_hgrid_face_nodes'), faces),
        'depth': ('nSCHISM_hgrid_node', depth),
        'elev': (('time', 'nSCHISM_hgrid_node'), elev),
        'zcor': (('time', 'nSCHISM_hgrid_node', 'nSCHISM_vgrid_layers'), zcor),
        'hvel': (('time', 'nSCHISM_hgrid_node', 'nSCHISM_vgrid_layers', 'two'), hvel),
        }, coords={'time': pd.date_range('2024-01-01', periods=nt, freq='1h')})
    filename = tmp_path / 'schout_1.nc'
    ds.to_netcdf(filename)
    return filename


def test_true_outline_with_island(schism_file):
    r = reader_schism_native.Reader(schism_file, proj4=proj4_nztm, use_3d=True)
    assert r.use_true_outline
    # in mesh, in island, outside of mesh
    x = np.array([1700000 + 200 * 5, 1700000 + 200 * 12, 1700000 + 200 * 40])
    y = np.array([5450000 + 200 * 5, 5450000 + 200 * 10, 5450000 + 200 * 5])
    np.testing.assert_array_equal(r.covers_positions(x, y), [True, False, False])


@pytest.mark.parametrize('buffer', [10000., 300.])
def test_3d_interpolation_frame(schism_file, buffer):
    """Same results with 3D KDtree built from nodes around elements, as with all nodes"""
    r = reader_schism_native.Reader(schism_file, proj4=proj4_nztm, use_3d=True,
                                    KDtree_3d_buffer=buffer)
    rng = np.random.default_rng(1)
    x = 1700000 + 200 * rng.uniform(2, 8, 50)
    y = 5450000 + 200 * rng.uniform(2, 8, 50)
    z = -rng.uniform(0, 20, 50)
    variables = ['x_sea_water_velocity', 'y_sea_water_velocity']
    data = r.get_variables(variables, r.times[1], x, y, z)
    block = reader_schism_native.ReaderBlockUnstruct(
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()},
        KDtree=r.reader_KDtree, x_particles=x, y_particles=y, KDtree_3d_buffer=buffer)
    block_all = reader_schism_native.ReaderBlockUnstruct(data, KDtree=r.reader_KDtree)
    small_frame = buffer < 1000  # otherwise frame covers whole (small) mesh
    if small_frame:
        assert len(block.id_frame_3d) < len(block.x_3d)
    # second set of positions moved away from frame, to use KDtree of all nodes
    for xp in [x, x + 200 * rng.uniform(0, 15, 50)]:
        env, _ = block.interpolate(xp, y, z, variables, None, 50)
        env_all, _ = block_all.interpolate(xp, y, z, variables, None, 50)
        for var in variables:
            np.testing.assert_array_equal(env[var], env_all[var])
    if small_frame:
        assert block.block_KDtree_3d_all is not None
