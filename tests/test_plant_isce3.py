import os
import pytest
import plant
from plant_isce3 import (geocode, runconfig, info, util, interpolate_dem,
                         topo)

NISAR_RSLC_PATH = 'data/envisat.h5'
DEM_PATH = 'data/constant_height.vrt'

GEOCODED_TIFF_PATH = 'output_data/gcov.tif'
NISAR_GCOV_RUNCONFIG = 'output_data/gcov.yaml'
NISAR_GCOV_PATH = 'output_data/gcov.h5'
RSLC_ORBIT_KML = 'output_data/rslc_orbit.kml'
GCOV_ORBIT_KML = 'output_data/gcov_orbit.kml'
INTERPOLATED_DEM_PATH = 'output_data/interpolated_dem.tif'
TOPO_DIR = 'output_data/topo_dir'


def test_plant_isce3_info():
    info(NISAR_RSLC_PATH)


def test_plant_isce3_util():
    util(NISAR_RSLC_PATH, orbit_kml=RSLC_ORBIT_KML,
         force=True)
    assert os.path.isfile(RSLC_ORBIT_KML)


def test_plant_isce3_topo():
    topo(NISAR_RSLC_PATH, dem=DEM_PATH,
         output_directory=TOPO_DIR)


def test_plant_isce3_geocode():
    geocode(NISAR_RSLC_PATH, dem=DEM_PATH,
            output_file=GEOCODED_TIFF_PATH, force=True)
    assert os.path.isfile(GEOCODED_TIFF_PATH)


def test_plant_isce3_gcov():
    runconfig(NISAR_RSLC_PATH, dem=DEM_PATH,
              sas_output_file=NISAR_GCOV_PATH,
              output_file=NISAR_GCOV_RUNCONFIG,
              force=True)
    assert os.path.isfile(NISAR_GCOV_RUNCONFIG)

    plant.execute(f'gcov.py {NISAR_GCOV_RUNCONFIG} --no-log')
    assert os.path.isfile(NISAR_GCOV_PATH)

    util(NISAR_GCOV_PATH, orbit_kml=GCOV_ORBIT_KML,
         force=True)
    assert os.path.isfile(GCOV_ORBIT_KML)

    info(NISAR_GCOV_PATH)
