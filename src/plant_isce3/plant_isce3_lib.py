import os
import sys
import time
import datetime
import json
import gc
import configparser
import geopandas as gpd

import plant_isce3
import importlib
from collections.abc import Sequence
from plant_isce3.readers import open_product
from nisar.workflows.geogrid import _grid_size
import numpy as np
import isce3
from osgeo import osr, gdal, gdal_array

import shapely.wkt

from plant_isce3.readers.orbit import load_orbit_from_xml

import plant

LIST_OF_SUPPORTED_NISAR_PRODUCTS = [
    'RRSD', 'RSLC', 'SLC', 'GCOV', 'GSLC', 'STATIC']
LIST_OF_NISAR_INSTRUMENTS = ['LSAR', 'SSAR']

DEFAULT_ISCE3_TEMPORARY_FORMAT = 'GTiff'


def add_arguments(parser,
                  product_type=0,

                  abs_cal_factor=0,
                  burst_ids=0,

                  track_number=0,
                  orbit_direction=0,
                  frame_number=0,
                  cycle_number=0,
                  track_frame_db=0,

                  geocode_cov_options=0,

                  data_interp_method=0,
                  dem_interp_method=0,
                  flag_upsample_radar_grid=0,
                  out_off_diag_terms=0,
                  geogrid_upsampling=0,
                  exponent=0,
                  out_geo_rdr=0,
                  out_geo_dem=0,
                  out_geo_nlooks=0,
                  out_geo_rtc=0,
                  out_geo_rtc_gamma0_to_sigma0=0,
                  output_rtc=0,
                  memory_mode=0,
                  min_block_size=0,
                  max_block_size=0,
                  clip_min=0,
                  clip_max=0,
                  min_nlooks=0,
                  geo2rdr_threshold=0,
                  geo2rdr_num_iter=0,

                  nlooks_by_frequency=0,
                  nlooks_x_a=0,
                  nlooks_y_a=0,
                  nlooks_x_b=0,
                  nlooks_y_b=0,

                  epsg=0,
                  frequency=0,
                  input_raster=0,

                  rtc_options=0,
                  input_rtc=0,
                  terrain_correction_type=0,
                  rtc_upsampling=0,
                  rtc_min_value_db=0,
                  input_terrain_radiometry=0,
                  output_terrain_radiometry=0,

                  native_doppler_grid=0,
                  orbit_files=0,

                  tec_files=0):

    if product_type:
        parser.add_argument('--product-type',
                            choices=['GCOV', 'GSLC', 'STATIC', 'RSLC', 'RRSD'],
                            type=str,
                            dest='product_type',
                            help='Product type')

    if abs_cal_factor:
        parser.add_argument('--abs-cal-factor',
                            '--abs-calibration-factor',
                            '--calibration-factor',
                            dest='abs_cal_factor',
                            type=float,
                            required=abs_cal_factor == 2,
                            help='Absolute calibration factor')

    if burst_ids:
        parser.add_argument('--burst-id',
                            '--burst-ids',
                            dest='burst_ids',
                            nargs='+',
                            type=str,
                            required=burst_ids == 2,
                            help=('Sentinel-1 burst IDs (only applicable for'
                                  ' Sentinel-1 datasets).'))
    if track_number:
        parser.add_argument('--track',
                            '--track-number',
                            type=int,
                            dest='track_number',
                            help='Track number')

    if orbit_direction:
        parser.add_argument('--orbit-direction',
                            '--orbit-pass-direction',
                            choices=['Ascending', 'Descending'],
                            type=str,
                            dest='orbit_direction',
                            help='Orbit pass direction for filename filtering')

    if frame_number:
        parser.add_argument('--frame',
                            '--frame-number',
                            type=int,
                            dest='frame_number',
                            help='Frame number')

    if cycle_number:
        parser.add_argument('--cycle',
                            '--cycle-number',
                            type=int,
                            dest='cycle_number',
                            help='Cycle number')

    if track_frame_db:
        parser.add_argument('--tfdb',
                            '--track-frame-db',
                            '--track-frame-database',
                            type=str,
                            dest='track_frame_db',
                            help='Track-frame database (TFDB) file')

    if (geocode_cov_options or
            data_interp_method or
            dem_interp_method or
            flag_upsample_radar_grid or
            out_off_diag_terms or
            geogrid_upsampling or
            exponent or
            out_geo_rdr or
            out_geo_dem or
            out_geo_nlooks or
            out_geo_rtc or
            out_geo_rtc_gamma0_to_sigma0 or
            output_rtc or
            memory_mode or
            min_block_size or
            max_block_size or
            clip_min or
            clip_max or
            min_nlooks or
            geo2rdr_threshold or
            geo2rdr_num_iter):
        geocode_cov_group = parser.add_argument_group(
            plant.PARSER_GROUP_SEPARATOR +
            'ISCE3 GeocodeCov arguments')

    if geocode_cov_options or data_interp_method:
        geocode_cov_group.add_argument(
            "--data-interp-method",
            dest="data_interp_method",
            type=str,

            required=data_interp_method == 2,
            help=(
                "Data interpolation method. Options:"
                " sinc, bilinear, bicubic, nearest,"
                " biquintic"
            ),
        )

    if geocode_cov_options or dem_interp_method:
        geocode_cov_group.add_argument(
            "--dem-interp-method",
            dest="dem_interp_method",
            type=str,
            default="biquintic",
            required=dem_interp_method == 2,
            help=(
                "DEM interpolation method. Options:"
                " sinc, bilinear, bicubic, nearest,"
                " biquintic"
            ),
        )

    if geocode_cov_options or flag_upsample_radar_grid:
        geocode_cov_group.add_argument(
            "--double-radar-grid-sampling",
            "--upsample-radar-grid",
            dest="flag_upsample_radar_grid",
            default=None,
            action="store_true",
            required=flag_upsample_radar_grid == 2,
            help="Double radar grid sampling.",
        )

    if geocode_cov_options or out_off_diag_terms:
        geocode_cov_group.add_argument(
            "--out-off-diag-terms",
            "--out-off-diagonal-terms",
            dest="out_off_diag_terms",
            type=str,
            required=out_off_diag_terms == 2,
            help="Output off-diagonal terms.",
        )

    if geocode_cov_options or geogrid_upsampling:
        geocode_cov_group.add_argument(
            "--upsampling",
            dest="geogrid_upsampling",
            type=float,
            required=geogrid_upsampling == 2,
            help="Geogrid upsample factor.",
        )

    if geocode_cov_options or exponent:
        geocode_cov_group.add_argument(
            "--exponent",
            dest="exponent",
            type=int,
            required=exponent == 2,
            help="Exponent for geocoding."
        )

    if geocode_cov_options or out_geo_rdr:
        geocode_cov_group.add_argument(
            "--out-geo-rdr",
            dest="out_geo_rdr",
            type=str,
            required=out_geo_rdr == 2,
            help="Output geo rdr file",
        )

    if geocode_cov_options or out_geo_dem:
        geocode_cov_group.add_argument(
            "--out-geo-dem",
            dest="out_geo_dem",
            type=str,
            required=out_geo_dem == 2,
            help="Output interpolated DEM file",
        )

    if geocode_cov_options or out_geo_nlooks:
        geocode_cov_group.add_argument(
            "--out-geo-nlooks",
            dest="out_geo_nlooks",
            type=str,
            required=out_geo_nlooks == 2,
            help="Output geo nlooks file",
        )

    if geocode_cov_options or out_geo_rtc:
        geocode_cov_group.add_argument(
            "--out-geo-rtc",
            dest="out_geo_rtc",
            type=str,
            required=out_geo_rtc == 2,
            help="Output geo RTC file",
        )

    if geocode_cov_options or out_geo_rtc_gamma0_to_sigma0:
        geocode_cov_group.add_argument(
            "--out-geo-rtc-gamma-to-sigma",
            "--out-geo-rtc-anf-gamma-to-sigma",
            "--out-geo-rtc-gamma0-to-sigma0",
            dest="out_geo_rtc_gamma0_to_sigma0",
            type=str,
            required=out_geo_rtc_gamma0_to_sigma0 == 2,
            help="Output geo RTC ANF to sigma0 file",
        )

    if geocode_cov_options or output_rtc:
        geocode_cov_group.add_argument(
            "--out-rtc",
            "--output-rtc",
            dest="output_rtc",
            type=str,
            required=output_rtc == 2,
            help="Output RTC ANF file (in slant-range)",
        )

    if geocode_cov_options or memory_mode:
        geocode_cov_group.add_argument(
            "--memory-mode",
            dest="memory_mode",
            type=str,
            choices=[
                "auto",
                "single-block",
                "blocks-geogrid",
                "blocks-geogrid-and-radargrid",
            ],
            required=memory_mode == 2,
            help="Memory mode",
        )

    if geocode_cov_options or min_block_size:
        geocode_cov_group.add_argument(
            "--min-block-size",
            type=int,
            dest="min_block_size",
            required=min_block_size == 2,
            help="Minimum block size in Bytes",
        )

    if geocode_cov_options or max_block_size:
        geocode_cov_group.add_argument(
            "--max-block-size",
            type=int,
            dest="max_block_size",
            required=max_block_size == 2,
            help="Maximum block size in Bytes",
        )

    if geocode_cov_options or clip_min:
        geocode_cov_group.add_argument(
            "--clip-min",
            type=float,
            dest="clip_min",
            required=clip_min == 2,
            help="Clip (limit) min output values",
        )

    if geocode_cov_options or clip_max:
        geocode_cov_group.add_argument(
            "--clip-max",
            type=float,
            dest="clip_max",
            required=clip_max == 2,
            help="Clip (limit) max output values",
        )

    if geocode_cov_options or min_nlooks:
        geocode_cov_group.add_argument(
            "--nlooks-min",
            "--min-nlooks",
            type=float,
            dest="min_nlooks",
            required=min_nlooks == 2,
            help="Minimum number of looks. Geogrid data"
            " below this limit will be set to NaN.",
        )

    if geocode_cov_options or geo2rdr_threshold:
        geocode_cov_group.add_argument(
            "--geo2rdr-threshold",
            type=float,

            dest="geo2rdr_threshold",
            help="Range convergence threshold for geo2rdr",
        )

    if geocode_cov_options or geo2rdr_num_iter:
        geocode_cov_group.add_argument(
            "--geo2rdr-num-iter",
            "--geo2rdr-numiter",
            type=float,

            dest="geo2rdr_num_iter",
            required=geo2rdr_num_iter == 2,
            help="Maximum number of iterations for geo2rdr",
        )

    if nlooks_by_frequency or nlooks_x_a:
        parser.add_argument('--nlooks-x-freq-a',
                            '--nlooks-x-a',
                            type=int,
                            help=('Number of looks in the X direction'
                                  ' for frequency A (when available)'),
                            dest='nlooks_x_a')

    if nlooks_by_frequency or nlooks_y_a:
        parser.add_argument('--nlooks-y-freq-a',
                            '--nlooks-y-a',
                            type=int,
                            help=('Number of looks in the Y direction'
                                  ' for frequency A (when available)'),
                            dest='nlooks_y_a')

    if nlooks_by_frequency or nlooks_x_b:
        parser.add_argument('--nlooks-x-freq-b',
                            '--nlooks-x-b',
                            type=int,
                            help=('Number of looks in the X direction'
                                  ' for frequency B (when available)'),
                            dest='nlooks_x_b')

    if nlooks_by_frequency or nlooks_y_b:
        parser.add_argument('--nlooks-y-freq-b',
                            '--nlooks-y-b',
                            type=int,
                            help=('Number of looks in the Y direction'
                                  ' for frequency B (when available)'),
                            dest='nlooks_y_b')

    if epsg:
        parser.add_argument(
            "--epsg",
            dest="epsg",
            type=int,
            required=epsg == 2,
            help="EPSG code for output grids.",
        )

    if frequency:
        parser.add_argument(
            "--frequency",
            dest="frequency",
            default=None,
            type=str,
            required=frequency == 2,
            help='Frequency band, either "A" or "B"',
        )

    if input_raster:
        parser.add_argument(
            "--input-raster",
            dest="input_raster",
            type=str,
            required=input_raster == 2,
            help="Input raster.",
        )

    if (rtc_options or
            input_rtc or
            terrain_correction_type or
            rtc_upsampling or
            rtc_min_value_db or
            input_terrain_radiometry or
            output_terrain_radiometry):

        rtc_group = parser.add_argument_group(
            plant.PARSER_GROUP_SEPARATOR +
            'ISCE3 RTC arguments')

    if rtc_options or input_rtc:
        rtc_group.add_argument(
            "--input-rtc",
            dest="input_rtc",
            type=str,
            required=input_rtc == 2,
            help="Input RTC area factor.",
        )

    if rtc_options or terrain_correction_type:
        rtc_group.add_argument(
            "--terrain",
            "--terrain-type",
            "--rtc",
            dest="terrain_correction_type",
            type=str,
            help="type of radiometric terrain correction: "
            "'gamma-naught-david-small', "
            "'gamma-naught-area-projection' "
            "(default: %(default)s)",
            required=terrain_correction_type == 2,
            default="gamma-naught-area-projection",
        )

    if rtc_options or rtc_upsampling:
        rtc_group.add_argument(
            "--rtc-upsampling",
            dest="rtc_upsampling",
            type=float,
            required=rtc_upsampling == 2,
            help="RTC geogrid upsample factor.",
        )

    if rtc_options or rtc_min_value_db:
        rtc_group.add_argument(
            "--rtc-min-value-db",
            dest="rtc_min_value_db",
            default=-30,
            type=float,
            required=rtc_min_value_db == 2,
            help=("RTC min. value in dB. -1 for disabled."
                  " Default: -30 dB."),
        )

    if rtc_options or input_terrain_radiometry:
        rtc_group.add_argument(
            "--input-radiometry",
            "--input-terrain-radiometry",
            dest="input_terrain_radiometry",
            type=str,
            required=input_terrain_radiometry == 2,
            help=("Input data radiometry. Options:"
                  "beta0 or sigma0-ellipsoid"),
        )

    if rtc_options or output_terrain_radiometry:
        rtc_group.add_argument(
            "--output-radiometry",
            "--output-terrain-radiometry",
            dest="output_terrain_radiometry",
            type=str,
            required=output_terrain_radiometry == 2,
            help=("Output data radiometry. Options:"
                  "sigma-naught or gamma-naught"),
        )

    if native_doppler_grid:
        parser.add_argument(
            "--native-doppler-grid",
            dest="native_doppler_grid",
            default=False,
            action="store_true",
            required=native_doppler_grid == 2,
            help=("Consider native Doppler grid (skewed"
                  " geometry)"),
        )

    if orbit_files:
        parser.add_argument(
            "--orbit",
            "--orbit-file",
            "--orbit-files",
            dest="orbit_files",
            nargs="+",
            type=str,
            required=orbit_files == 2,
            help="Orbit file.",
        )

    if tec_files:
        parser.add_argument(
            "--tec",
            "--tec-file",
            "--tec-files",
            dest="tec_files",
            nargs="+",
            type=str,
            required=tec_files == 2,
            help="Total electron content (TEC) file",
        )


def is_nisar_format(h5_obj):

    if 'mission_name' in h5_obj.attrs:
        mission_name = h5_obj.attrs['mission_name']
        try:
            if not isinstance(mission_name, str):
                mission_name = mission_name.decode()
            if mission_name == 'NISAR':
                return True
        except BaseException:
            pass

    return get_nisar_product_type(h5_obj) is not None


def get_nisar_identification_scalar(h5_obj, scalar_name, default_value=None):

    for instrument in LIST_OF_NISAR_INSTRUMENTS:
        product_type_key = (f'/science/{instrument}/'
                            f'identification/{scalar_name}')
        if product_type_key in h5_obj:
            product_type = h5_obj[product_type_key][()]
            if isinstance(product_type, (bytes, np.bytes_)):
                product_type = product_type.decode()
            return product_type
    return default_value


def get_nisar_product_instrument_name(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'instrumentName')


def get_nisar_product_type(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'productType')


def get_nisar_product_bounding_polygon(h5_obj, flag_as_list=False,
                                       flag_as_shapely_obj=False):

    if flag_as_list and flag_as_shapely_obj:
        print('ERROR only one option is allowed:'
              ' `flag_as_shapely_obj` or `flag_as_list`')

    polygon_str = \
        get_nisar_identification_scalar(h5_obj, 'boundingPolygon')

    if not flag_as_list or flag_as_shapely_obj:
        return polygon_str

    if flag_as_shapely_obj:
        return shapely.wkt.loads(polygon_str)

    polygon_str = polygon_str.replace('POLYGON', '')
    polygon_str_ref = ''
    while polygon_str_ref != polygon_str:
        polygon_str_ref = polygon_str
        polygon_str = polygon_str.replace('(', '')
    polygon_str_ref = ''
    while polygon_str_ref != polygon_str:
        polygon_str_ref = polygon_str
        polygon_str = polygon_str.replace(')', '')
    polygon = polygon_str.split(',')
    polygon = [p.strip().split(' ') for p in polygon]

    return polygon


def get_nisar_product_level(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'productLevel')


def get_nisar_orbit_direction(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'orbitPassDirection')


def get_nisar_granule_id(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'granuleId')


def get_nisar_product_absolute_orbit_number(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'absoluteOrbitNumber')


def get_nisar_product_mission_id(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'missionId')


def get_nisar_product_cycle_number(h5_obj, flag_no_offset=False):

    absolute_orbit_number = get_nisar_product_absolute_orbit_number(h5_obj)

    if absolute_orbit_number is None:
        return

    mission_id = get_nisar_product_mission_id(h5_obj)

    if mission_id == 'NISAR' and not flag_no_offset:

        offset = 792
    else:
        offset = 1

    cycle_number = (absolute_orbit_number - offset) // 173 + 1

    return cycle_number


def get_nisar_product_track_number(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'trackNumber')


def get_nisar_product_frame_number(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'frameNumber')


def get_nisar_product_is_mixed_mode(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'isMixedMode')


def get_nisar_product_is_full_frame(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'isFullFrame')


def get_nisar_product_zero_doppler_start_time(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'zeroDopplerStartTime')


def get_nisar_product_zero_doppler_end_time(h5_obj):

    return get_nisar_identification_scalar(h5_obj, 'zeroDopplerEndTime')


def get_input_l0b_granules_from_rslc(h5_obj):

    input_l0b_granules = \
        h5_obj['/science/LSAR/RSLC/metadata/'
               'processingInformation/inputs/l0bGranules'][()]
    input_l0b_granules = [d.decode() for d in input_l0b_granules]
    if len(input_l0b_granules) == 1:
        input_l0b_granules = input_l0b_granules[0]

    return input_l0b_granules


def multilook_isce3(input_raster_file, output_file,
                    nlooks_y, nlooks_x,
                    transform_square=False,
                    block_nlines=4096,
                    output_format=None,
                    metadata_dict=None,
                    verbose=True):

    input_raster = isce3.io.Raster(input_raster_file)

    if nlooks_y is None:
        nlooks_y = 1

    if nlooks_x is None:
        nlooks_x = 1

    width = input_raster.width
    length = input_raster.length

    width_ml = int(width // nlooks_x)
    length_ml = int(length // nlooks_y)

    exponent = 2 if transform_square else 0

    if exponent % 2 == 0:
        output_dtype = gdal.GDT_Float32
    else:
        output_dtype = gdal.GDT_CFloat32
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    nbands = input_raster.num_bands

    isce3_format = get_isce3_temporary_format(output_file,
                                              output_format=output_format)

    output_raster = isce3.io.Raster(output_file,

                                    width_ml,
                                    length_ml,
                                    nbands,
                                    output_dtype,
                                    isce3_format)

    if verbose:
        print('block number of lines:', block_nlines)

    if block_nlines is not None and plant.isvalid(block_nlines):
        block_nlines = int(np.ceil(float(block_nlines) / nlooks_y) *
                           nlooks_y)

        n_blocks = int(np.ceil(float(length) / block_nlines))
    else:
        block_nlines = length
        n_blocks = 1

    if verbose:
        print('block number of lines (rounded to nlooks az):', block_nlines)

    for band in range(nbands):

        with plant.PrintProgress(n_blocks) as progress_obj:

            for block in range(n_blocks):
                progress_obj.print_progress(block)

                start_line = block_nlines * block
                end_line = min([block_nlines * (block + 1),
                                length + 1])
                block_array = input_raster.get_block(
                    key=np.s_[start_line:end_line, :],
                    band=band + 1)

                if exponent > 1:
                    block_array = np.absolute(block_array) ** 2

                if nlooks_y == 1 and nlooks_x == 1:
                    multilooked_image = block_array

                else:

                    is_finite_array = np.isfinite(block_array)

                    block_array[~is_finite_array] = 0

                    multilooked_image = isce3.signal.multilook_summed(
                        block_array,
                        int(nlooks_y), int(nlooks_x))
                    multilooked_image_is_finite = \
                        isce3.signal.multilook_summed(
                            is_finite_array, int(nlooks_y), int(nlooks_x))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        multilooked_image = (multilooked_image /
                                             multilooked_image_is_finite)
                    multilooked_image[multilooked_image_is_finite == 0] = \
                        np.nan

                start_line_ml = int(start_line // nlooks_y)
                end_line_ml = int(end_line // nlooks_y)
                output_raster.set_block(
                    key=np.s_[start_line_ml:end_line_ml, :],
                    value=multilooked_image,
                    band=band + 1)

    del input_raster
    del output_raster

    input_gdal_ds = gdal.Open(input_raster_file)
    geotransform = input_gdal_ds.GetGeoTransform()
    if geotransform is None:
        return

    geotransform = list(geotransform)
    projection = input_gdal_ds.GetProjection()
    input_gdal_ds.FlushCache()
    input_gdal_ds.Close()

    del input_gdal_ds

    output_gdal_ds = gdal.Open(output_file, gdal.GA_Update)
    plant_geogrid_obj = plant.get_coordinates(geotransform=geotransform,
                                              width=width,
                                              length=length,
                                              projection=projection)
    if plant_geogrid_obj.has_valid_coordinates():

        geotransform[1] = geotransform[1] * nlooks_x
        geotransform[5] = geotransform[5] * nlooks_y

        output_gdal_ds.SetGeoTransform(geotransform)
        output_gdal_ds.SetProjection(projection)

    if metadata_dict is not None:
        output_gdal_ds.SetMetadata(metadata_dict)

    output_gdal_ds.FlushCache()
    output_gdal_ds.Close()

    del output_gdal_ds


def apply_slc_corrections(burst_in,

                          flag_output_complex: bool = False,
                          flag_thermal_correction: bool = True,
                          flag_apply_abs_rad_correction: bool = True,
                          clip_min=0):

    temp_vrt = plant.get_temporary_file(ext='vrt')
    burst_in.slc_to_vrt_file(temp_vrt)
    slc_gdal_ds = gdal.Open(temp_vrt)
    arr_slc_from = slc_gdal_ds.ReadAsArray()

    if flag_thermal_correction:
        print('applying thermal noise correction to burst SLC')
        corrected_image = (np.abs(arr_slc_from) ** 2 -
                           burst_in.thermal_noise_lut)

        min_backscatter = clip_min
        max_backscatter = None
        corrected_image = np.clip(corrected_image, min_backscatter,
                                  max_backscatter)

    else:
        corrected_image = np.abs(arr_slc_from) ** 2

    if flag_apply_abs_rad_correction:
        print('applying absolute radiometric correction to burst'
              ' SLC')
        corrected_image = \
            corrected_image / burst_in.burst_calibration.beta_naught ** 2

    if flag_output_complex:
        factor_mag = np.sqrt(corrected_image) / np.abs(arr_slc_from)
        factor_mag[np.isnan(factor_mag)] = 0.0
        corrected_image = arr_slc_from * factor_mag

    return corrected_image


def get_isce3_raster(raster_file, *args, **kwargs):
    expanded_filename = plant.get_filename(raster_file)
    if expanded_filename is None:
        return isce3.io.Raster(raster_file, *args, **kwargs)
    return isce3.io.Raster(expanded_filename, *args, **kwargs)


def get_attribute(attribute_name, precedence_value, plant_script_obj):
    if precedence_value is not None:
        return precedence_value

    return getattr(plant_script_obj, attribute_name, None)


def get_isce3_temporary_format(output_file, output_format=None):

    if output_format is not None and 'tif' in output_format.lower():
        output_format = 'GTiff'
        if output_format in plant.OUTPUT_FORMAT_MAP.keys():
            output_format = plant.OUTPUT_FORMAT_MAP[output_format]
        return output_format

    if not output_file:
        return DEFAULT_ISCE3_TEMPORARY_FORMAT

    _, extension = os.path.splitext(output_file)

    extension = extension.lower()
    if extension and extension.startswith('.'):
        extension = extension[1:]
    if (extension == 'tif' or extension == 'tiff'):
        output_format = 'GTiff'
    elif (extension == 'bin'):
        output_format = 'ENVI'

    else:
        output_format = DEFAULT_ISCE3_TEMPORARY_FORMAT
    if output_format in plant.OUTPUT_FORMAT_MAP.keys():
        output_format = plant.OUTPUT_FORMAT_MAP[output_format]
    return output_format


def get_files_from_s3_bucket(my_bucket, s3_prefix,
                             extension=None, verbose=False):

    if verbose:
        print('s3_prefix:', s3_prefix)

    if s3_prefix:
        files_iterator = my_bucket.objects.filter(Prefix=s3_prefix)
    else:
        files_iterator = my_bucket.objects.all()

    file_list = []
    for _, objects in enumerate(files_iterator):

        if verbose:
            print('objets.key:', objects.key)

        file_path = objects.key
        if extension and not file_path.endswith(extension):
            continue

        if verbose:
            print('file_path:', file_path, file_path.__class__)
        file_list.append(objects.key)

    return file_list


class PlantIsce3Sensor():

    def __init__(self, plant_script_obj=None, input_file=None,
                 orbit_files=None, tec_files=None, burst_ids=None,
                 verbose=True):

        self.verbose = verbose
        self.plant_script_obj = plant_script_obj

        self.input_file = None
        if self.input_file is None:
            self.input_file = plant_script_obj.input_file
        if not self.input_file:
            print(f'ERROR invalid input file: "{input_file}"')

        self.orbit_files = get_attribute('orbit_files', orbit_files,
                                         plant_script_obj)

        self.tec_files = get_attribute('tec_files', tec_files,
                                       plant_script_obj)

        self.burst_ids = get_attribute('burst_ids', burst_ids,
                                       plant_script_obj)

        self.load_product()

    def load_product(self):

        if self.input_file.endswith('.h5'):
            with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
                if not is_nisar_format(h5_obj):
                    print(f'ERROR file not recognized: {self.input_file}')
                    return

                product_type = get_nisar_product_type(h5_obj)

                if product_type is None:
                    print('ERROR cannot determine NISAR product type for file:'
                          f' {self.input_file}')
                    return

                if product_type not in LIST_OF_SUPPORTED_NISAR_PRODUCTS:
                    print('ERROR unsupported NISAR product type:'
                          f' {product_type}. List of supported NISAR products:'
                          f' {LIST_OF_SUPPORTED_NISAR_PRODUCTS}')
                    return

            self.sensor_name = 'NISAR'
            self.nisar_product_obj = open_product(self.input_file)
            self.frequency = self.get_frequency()
            return

        if self.input_file.endswith('.zip'):
            self.sensor_name = 'Sentinel-1'
            self.load_sentinel_1_bursts()
            return

        print(f'ERROR file not recognized: {self.input_file}')

    def get_frequency(self):
        if (self.sensor_name != 'NISAR'):
            return

        if self.plant_script_obj is not None:
            frequency = getattr(self.plant_script_obj, 'frequency', None)
            if frequency is not None:
                return frequency

        frequency_list = list(
            self.nisar_product_obj.polarizations.keys())

        if len(frequency_list) == 0:
            return

        frequency = frequency_list[0]

        return frequency

    def load_sentinel_1_bursts(self):
        if (self.sensor_name == 'NISAR'):
            print(f'ERROR cannot load bursts from a {self.sensor_name}'
                  ' dataset')
            return
        from s1reader.s1_reader import load_bursts

        orbit_file = self.get_orbit_file()

        if self.verbose:
            print('burst ID(s):', self.burst_ids)
        if self.burst_ids is None:
            subswath_number_list = [1, 2, 3]

        else:
            subswath_number_list = []
            for burst_id in self.burst_ids:
                iw_index = burst_id.upper().find('IW')
                if iw_index < 0 and self.verbose:
                    print(f'ERROR invalid burst ID: {burst_id}')
                subswath_number = int(burst_id[iw_index + 2])
                subswath_number_list.append(subswath_number)
        if self.verbose:
            print('IW subswath(s):', subswath_number_list)

        self.burst_dict = {}

        self.pol_list = []

        for subswath_number in subswath_number_list:

            for pol in ['VV', 'VH', 'HH', 'HV']:
                burst_list_this_pol = None
                try:
                    burst_list_this_pol = load_bursts(
                        self.input_file, orbit_file,
                        subswath_number, pol, flag_apply_eap=False)
                except ValueError:
                    continue

                if pol not in self.pol_list:
                    self.pol_list.append(pol)

                for burst in burst_list_this_pol:

                    burst_id = str(burst.burst_id)

                    if (self.burst_ids is not None and
                            burst_id.lower() not in self.burst_ids):
                        continue

                    if burst_id not in self.burst_dict.keys():
                        self.burst_dict[burst_id] = {}
                    self.burst_dict[burst_id][pol] = burst

        if len(self.burst_dict) == 0:
            print(f'ERROR reading file {self.input_file} with orbit file(s)'
                  f' {self.orbit_files}')

        if self.verbose:
            print('polarizations:', self.pol_list)
            print('number of bursts:', len(self.burst_dict))

    @property
    def burst(self):
        if len(self.burst_dict) != 1:
            print('ERROR the dataset contains multiple bursts:'
                  f' {self.burst_dict.keys()}. Please select a burst.')
            return

        burst_id = list(self.burst_dict.keys())[0]
        first_pol = list(self.burst_dict[burst_id])[0]

        return self.burst_dict[burst_id][first_pol]

    def get_sentinel_1_input_raster(self, input_raster=None,
                                    flag_transform_input_raster=None,
                                    flag_output_complex=False,
                                    flag_thermal_correction=True,
                                    flag_apply_abs_rad_correction=True,
                                    clip_min=0):

        if input_raster is not None:
            if flag_transform_input_raster is not False:
                flag_apply_transformation = \
                    self.plant_script_obj.plant_transform_obj.flag_apply_transformation()
                image_obj = self.plant_script_obj.read_image(input_raster)
            else:
                flag_apply_transformation = False
                image_obj = plant.read_image(input_raster)
            if flag_apply_transformation:
                temp_file = plant.get_temporary_file(append=True,
                                                     ext='vrt')

                for b in range(image_obj.nbands):
                    band = image_obj.get_band(band=b)
                    image_obj.set_band(band, band=b)
                print(f'*** creating temporary file: {temp_file}')
                self.plant_script_obj.save_image(image_obj, temp_file,
                                                 force=True,
                                                 output_format='VRT')
                input_raster = temp_file
            return input_raster

        if len(self.burst_dict) != 1:
            print('ERROR the dataset contains multiple bursts:'
                  f' {self.burst_dict.keys()}. Please select a burst.')
            return

        pol_array_list = []
        for burst_pol_dict in self.burst_dict.values():

            pol_list = list(burst_pol_dict.keys())
            burst = burst_pol_dict[pol_list[0]]

            corrected_image = plant_isce3.apply_slc_corrections(
                burst,

                flag_output_complex=flag_output_complex,
                flag_thermal_correction=flag_thermal_correction,
                flag_apply_abs_rad_correction=flag_apply_abs_rad_correction,
                clip_min=clip_min)

            pol_array_list.append(corrected_image)

        temp_all_pol_vrt = plant.get_temporary_file(ext='tif')
        plant.save_image(pol_array_list, output_file=temp_all_pol_vrt,
                         force=True)
        plant.append_temporary_file(temp_all_pol_vrt)
        return temp_all_pol_vrt

    def get_orbit(self):
        if self.sensor_name == 'NISAR':

            orbit_file = self.get_orbit_file()
            if orbit_file is not None:
                radar_grid = self.get_radar_grid()
                orbit = load_orbit_from_xml(orbit_file,
                                            radar_grid.ref_epoch)
                return orbit

            return self.nisar_product_obj.getOrbit()

        if self.sensor_name == 'Sentinel-1':
            return self.burst.orbit

        print(f'ERROR sensor not supported: {self.sensor_name}')

    def get_grid_doppler(self):
        if self.sensor_name == 'NISAR':
            return self.plant_script_obj.get_doppler_grid_lut_nisar(
                self.nisar_product_obj)

        if self.sensor_name == 'Sentinel-1':
            return self.plant_script_obj.get_doppler_grid_lut_s1(
                self.burst)

        print(f'ERROR sensor not supported: {self.sensor_name}')

    def get_doppler_centroid(self):
        if self.sensor_name == 'NISAR':
            return self.plant_script_obj.get_doppler_centroid_lut_nisar(
                self.nisar_product_obj)

        if self.sensor_name == 'Sentinel-1':
            return self.plant_script_obj.get_doppler_centroid_lut_s1(
                self.burst)

        print(f'ERROR sensor not supported: {self.sensor_name}')

    def get_radar_grid(self, frequency=None):
        if self.sensor_name == 'NISAR':
            if frequency is None:
                frequency = self.get_frequency()
            return self.nisar_product_obj.getRadarGrid(
                frequency)

        if self.sensor_name == 'Sentinel-1':
            return self.burst.as_isce3_radargrid()

        print(f'ERROR sensor not supported: {self.sensor_name}')

    def get_radar_grid_ml(self, frequency=None):
        radar_grid = self.get_radar_grid(frequency=frequency)

        radar_grid_ml = self.plant_script_obj.get_radar_grid_ml(
            radar_grid, frequency=frequency)

        return radar_grid_ml

    def get_tec_file(self):
        if not self.tec_files:
            return

        all_tec_files = []
        for tec_file_list in self.tec_files:
            all_tec_files.extend(plant.glob(os.path.expanduser(tec_file_list)))

        if len(all_tec_files) == 0:
            print(f'ERROR invalid TEC file: {self.tec_files}')
            return

        if len(all_tec_files) == 1:
            return self.tec_files[0]

        if self.verbose:
            print('A list of TEC files has been provided. Filtering'
                  ' list by date.')

        radar_grid = self.get_radar_grid()
        radargrid_ref_epoch = datetime.datetime.fromisoformat(
            radar_grid.ref_epoch.isoformat_usec())
        sensing_start = radargrid_ref_epoch + datetime.timedelta(
            seconds=radar_grid.sensing_start)
        sensing_stop = radargrid_ref_epoch + datetime.timedelta(
            seconds=radar_grid.sensing_stop)

        for tec_file in all_tec_files:

            with open(tec_file, 'r') as jin:
                imagen_dict = json.load(jin)
                num_utc = len(imagen_dict['utc'])
                tec_start = datetime.datetime.fromisoformat(
                    imagen_dict['utc'][0])
                tec_end = datetime.datetime.fromisoformat(
                    imagen_dict['utc'][-1])

            tec_margin_start = (sensing_start - tec_start).total_seconds()
            tec_margin_end = (tec_end - sensing_stop).total_seconds()

            minimum_margin_sec = ((tec_end - tec_start).total_seconds() /
                                  (num_utc - 1) / 2)

            if (tec_margin_start < minimum_margin_sec or
                    tec_margin_end < minimum_margin_sec) and self.verbose:
                print('    TEC does not cover data:', tec_file,
                      tec_margin_start, tec_margin_end)
                continue

            if self.verbose:
                print(f'Selected TEC file: {tec_file}')

            return tec_file

        print('ERROR not TEC file intersects with the input product')

    def get_orbit_file(self):
        if not self.orbit_files:
            return

        all_orbit_files = []
        for orbit_file_list in self.orbit_files:
            all_orbit_files.extend(plant.glob(os.path.expanduser(
                orbit_file_list)))

        if len(all_orbit_files) == 0:
            print(f'ERROR invalid orbit file: {self.orbit_files}')
            return

        if len(all_orbit_files) == 1:
            return self.orbit_files[0]

        if self.sensor_name == 'Sentinel-1':
            print('ERROR A list of orbit files has been provided. The option'
                  ' to filter the orbit files by date is only available'
                  ' for NISAR files. Please, profile a single orbit file.')
            return

        if self.verbose:
            print('A list of orbit files has been provided. Filtering'
                  ' list by date.')

        radar_grid = self.get_radar_grid()
        radargrid_ref_epoch = datetime.datetime.fromisoformat(
            radar_grid.ref_epoch.isoformat_usec())
        sensing_start = radargrid_ref_epoch + datetime.timedelta(
            seconds=radar_grid.sensing_start)
        sensing_stop = radargrid_ref_epoch + datetime.timedelta(
            seconds=radar_grid.sensing_stop)

        for orbit_file in all_orbit_files:

            orbit = load_orbit_from_xml(orbit_file, radar_grid.ref_epoch)

            orbit_start = datetime.datetime.fromisoformat(
                orbit.start_datetime.isoformat_usec())
            orbit_end = datetime.datetime.fromisoformat(
                orbit.end_datetime.isoformat_usec())

            orbit_margin_start = (sensing_start - orbit_start).total_seconds()
            orbit_margin_end = (orbit_end - sensing_stop).total_seconds()

            if orbit_margin_start < 0 or orbit_margin_end < 0 and self.verbose:
                print('    orbit does not cover data:', orbit_file,
                      orbit_margin_start, orbit_margin_end)
                continue

            if self.verbose:
                print(f'Selected orbit file: {orbit_file}')

            return orbit_file

        print('ERROR not orbit file intersects with the input product')

    def get_sentinel_1_epsg(self):
        if (self.sensor_name == 'NISAR'):
            print(f'ERROR cannot load bursts from a {self.sensor_name}'
                  ' dataset')
            return
        y_list = []
        x_list = []

        for burst_pol_dict in self.burst_dict.values():

            first_pol = list(burst_pol_dict.keys())[0]
            burst = burst_pol_dict[first_pol]
            y_list.append(burst.center.y)
            x_list.append(burst.center.x)
        y_mean = np.nanmean(y_list)
        x_mean = np.nanmean(x_list)
        if self.verbose:
            print(f'center position (Y, X): ({y_mean}, {x_mean})')
        center_epsg = point2epsg(x_mean, y_mean)
        if self.verbose:
            print(f'center position EPSG: {center_epsg}')
        assert 1024 <= center_epsg <= 32767

        self.plant_script_obj.epsg = center_epsg

        return center_epsg

    def generate_geogrids(self):

        plant_geogrid_obj = self.plant_script_obj.plant_geogrid_obj
        width = plant_geogrid_obj.width
        length = plant_geogrid_obj.length
        xmin = plant_geogrid_obj.x0
        xmax = plant_geogrid_obj.xf
        ymax = plant_geogrid_obj.y0
        ymin = plant_geogrid_obj.yf
        x_spacing = plant_geogrid_obj.step_x
        y_spacing = plant_geogrid_obj.step_y

        xmin_all_bursts = np.inf
        ymax_all_bursts = -np.inf
        xmax_all_bursts = -np.inf
        ymin_all_bursts = np.inf

        epsg = self.get_sentinel_1_epsg()

        geogrids_dict = {}
        for burst_pol_dict in self.burst_dict.values():
            first_pol = list(burst_pol_dict.keys())[0]
            burst = burst_pol_dict[first_pol]
            burst_id = str(burst.burst_id)

            radar_grid = burst.as_isce3_radargrid()
            orbit = burst.orbit

            geogrid_burst = None

            geogrid_burst = isce3.product.bbox_to_geogrid(
                radar_grid, orbit, isce3.core.LUT2d(), x_spacing,
                y_spacing, epsg)

            geogrid_snapped = geogrid_burst

            xmin_all_bursts = min([xmin_all_bursts, geogrid_snapped.start_x])
            ymax_all_bursts = max([ymax_all_bursts, geogrid_snapped.start_y])
            xmax_all_bursts = max([xmax_all_bursts,
                                   geogrid_snapped.start_x +
                                   geogrid_snapped.spacing_x *
                                   geogrid_snapped.width])
            ymin_all_bursts = min([ymin_all_bursts,
                                   geogrid_snapped.start_y +
                                   geogrid_snapped.spacing_y *
                                   geogrid_snapped.length])

            geogrids_dict[burst_id] = geogrid_snapped

        if xmin is None:
            xmin = xmin_all_bursts
        if ymax is None:
            ymax = ymax_all_bursts
        if xmax is None:
            xmax = xmax_all_bursts
        if ymin is None:
            ymin = ymin_all_bursts

        width = _grid_size(xmax, xmin, x_spacing)
        length = _grid_size(ymin, ymax, y_spacing)
        geogrid_all = isce3.product.GeoGridParameters(
            xmin, ymax, x_spacing, y_spacing,
            width, length, epsg)

        return geogrid_all, geogrids_dict

    def get_h5_dataset(self, path, *args, **kwargs):
        if (self.sensor_name != 'NISAR'):
            raise RuntimeError

        if not is_nisar_format(self.input_file):
            raise RuntimeError(f'ERROR file not recognized: {self.input_file}')

        with plant.h5py_file_wrapper(self.input_file, *args, **kwargs) as \
                h5_obj:
            ret = h5_obj[path][()]

        return ret

    def get_nisar_identification_scalar(self, scalar_name, default_value=None):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            scalar = get_nisar_identification_scalar(
                h5_obj, scalar_name, default_value=default_value)

        return scalar

    def get_nisar_product_instrument_name(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            instrument_name = get_nisar_product_instrument_name(h5_obj)
        return instrument_name

    def get_nisar_product_type(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            product_type = get_nisar_product_type(h5_obj)
        return product_type

    def get_nisar_product_bounding_polygon(self, flag_as_list=False):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            polygon = get_nisar_product_bounding_polygon(
                h5_obj, flag_as_list=flag_as_list)

        return polygon

    def get_nisar_product_level(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            product_level = get_nisar_product_level(h5_obj)
        return product_level

    def get_nisar_orbit_direction(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            orbit_direction = get_nisar_orbit_direction(h5_obj)
        return orbit_direction

    def get_nisar_granule_id(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            granule_id = get_nisar_granule_id(h5_obj)
        return granule_id

    def get_nisar_product_absolute_orbit_number(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            absolute_orbit_number = \
                get_nisar_product_absolute_orbit_number(h5_obj)
        return absolute_orbit_number

    def get_nisar_product_cycle_number(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            cycle_number = get_nisar_product_cycle_number(h5_obj)
        return cycle_number

    def get_nisar_product_track_number(self):

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            track_number = get_nisar_product_track_number(h5_obj)
        return track_number

    def get_nisar_product_frame_number(self):
        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            product_frame_number = get_nisar_product_frame_number(h5_obj)
        return product_frame_number

    def get_nisar_product_is_mixed_mode(self):

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            is_mixed_mode = get_nisar_product_is_mixed_mode(h5_obj)
        return is_mixed_mode

    def get_nisar_product_is_full_frame(self):

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            is_full_frame = get_nisar_product_is_full_frame(h5_obj)
        return is_full_frame

    def get_nisar_product_zero_doppler_start_time(self):

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            zero_doppler_start_time = \
                get_nisar_product_zero_doppler_start_time(h5_obj)
        return zero_doppler_start_time

    def get_nisar_product_zero_doppler_end_time(self):

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            zero_doppler_end_time = \
                get_nisar_product_zero_doppler_end_time(h5_obj)

        return zero_doppler_end_time


class PlantIsce3Script(plant.PlantScript):

    def __init__(self, *args, **kwargs):

        try:
            super().__init__(*args, plant_isce3=True, **kwargs)
        except TypeError:
            super().__init__(*args, **kwargs)

        if self.getattr2('input_file') is not None:
            ret_dict = plant.parse_filename(self.input_file)
            if 'driver' in ret_dict.keys() and ret_dict['driver'] == 'NISAR':
                self.input_file = ret_dict['filename']
                frequency_from_key = ret_dict['key']

                if (self.hasattr('frequency') and
                    self.frequency is not None and
                    frequency_from_key is not None and
                        self.frequency != frequency_from_key):
                    self.print('ERROR argument frequency ("{self.frequency}")'
                               ' differs from NISAR-driver frequency'
                               f' ("{frequency_from_key}")')
                    return

                elif (self.hasattr('frequency') and
                      frequency_from_key is not None):
                    self.frequency = frequency_from_key

    def load_product(self, verbose=True):

        return PlantIsce3Sensor(plant_script_obj=self, verbose=verbose)

    def update_output_format(self, ret_dict):
        for output_file in ret_dict.values():

            image_obj = plant.read_image(output_file)
            actual_output_format = image_obj.file_format

            if self.getattr2('output_format') is None:

                expected_output_format = plant.get_output_format(
                    output_file)
                if expected_output_format == actual_output_format:
                    continue

            self.save_image(
                output_file,
                output_file=output_file,

                force=True)

            if actual_output_format != 'ENVI':
                continue

            envi_header = plant.get_envi_header(output_file)
            if os.path.isfile(envi_header):
                os.remove(envi_header)

    def get_grids_ref(self, layer_name, frequency,
                      nisar_product_obj, image_obj,
                      valid_products=['GCOV', 'GSLC', 'STATIC']):
        if image_obj is not None:
            return image_obj
        if nisar_product_obj.productType not in valid_products:
            error_msg = (f'ERROR cannot save layer "{layer_name}" for'
                         ' product type'
                         f' "{nisar_product_obj.productType}".')
            print(error_msg)
            raise ValueError(error_msg)

        if nisar_product_obj.productType == 'STATIC':
            grid_path = (f'{nisar_product_obj.GridPath}/{layer_name}')
        else:
            grid_path = (f'{nisar_product_obj.GridPath}'
                         f'/frequency{frequency}/{layer_name}')
        image_ref = f'NETCDF:{self.input_file}:{grid_path}'

        return image_ref

    def update_geogrid(self, radar_grid, dem_raster=None, geo=None,
                       nisar_product_obj=None, orbit=None):

        flag_update_geo = geo is not None

        if geo is None:
            if nisar_product_obj is None:
                nisar_product_obj = open_product(self.input_file)

            if orbit is None:
                orbit = nisar_product_obj.getOrbit()

            geo = isce3.geocode.GeocodeFloat32()
            geo.orbit = orbit
            geo.ellipsoid = isce3.core.Ellipsoid()

        width = self.plant_geogrid_obj.width
        length = self.plant_geogrid_obj.length
        x0_orig = self.plant_geogrid_obj.x0
        y0_orig = self.plant_geogrid_obj.y0
        step_x = self.plant_geogrid_obj.step_x
        step_y = self.plant_geogrid_obj.step_y

        if width is None:
            width = -9999
        if length is None:
            length = -9999

        if x0_orig is None:
            x0_orig = np.nan
        if y0_orig is None:
            y0_orig = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_x):
            step_x = plant.m_to_deg_lon(30.)
        elif step_x is None:
            step_x = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_y):
            step_y = - plant.m_to_deg_lat(30.)
        elif step_y is None:
            step_y = np.nan

        print('*** x0:', x0_orig)
        print('*** y0:', y0_orig)
        print('*** step_x:', step_x)
        print('*** step_y:', step_y)
        print('*** length:', length)
        print('*** width:', width)

        geo.geogrid(x0_orig,
                    y0_orig,
                    step_x,
                    step_y,
                    width,
                    length,
                    self.epsg)

        if dem_raster is not None:
            geo.update_geogrid(radar_grid, dem_raster)

        print('*** x0:', self.plant_geogrid_obj.x0)
        print('*** xf:', self.plant_geogrid_obj.xf)
        print('*** y0:', self.plant_geogrid_obj.y0)
        print('*** yf:', self.plant_geogrid_obj.yf)
        print('*** step_x:', self.plant_geogrid_obj.step_x)
        print('*** step_y:', self.plant_geogrid_obj.step_y)
        print('*** length:', self.plant_geogrid_obj.length)
        print('*** width:', self.plant_geogrid_obj.width)
        print('*** self.epsg:', self.epsg)
        projection = plant.epsg_to_wkt(self.epsg)
        print('*** projection:', projection)

        print('*** x0 (from geo):', geo.geogrid_start_x)
        print('*** width (from geo):', geo.geogrid_width)
        print('*** y0 (from geo):', geo.geogrid_start_y)
        print('*** length (from geo):', geo.geogrid_length)
        print('*** step_x (from geo):', geo.geogrid_spacing_x)
        print('*** step_y (from geo):', geo.geogrid_spacing_y)

        plant_geogrid_from_geo_obj = plant.PlantGeogrid(
            y0=geo.geogrid_start_y,
            length=geo.geogrid_length,
            x0=geo.geogrid_start_x,
            width=geo.geogrid_width,
            step_x=geo.geogrid_spacing_x,
            step_y=geo.geogrid_spacing_y,
            projection=projection)

        print('*** x0 (from geo 2):', plant_geogrid_from_geo_obj.x0)
        print('*** xf (from geo 2):', plant_geogrid_from_geo_obj.xf)
        print('*** y0 (from geo 2):', plant_geogrid_from_geo_obj.y0)
        print('*** yf (from geo 2):', plant_geogrid_from_geo_obj.yf)
        print('*** step_x (from geo 2):', plant_geogrid_from_geo_obj.step_x)
        print('*** step_y (from geo 2):', plant_geogrid_from_geo_obj.step_y)

        self.plant_geogrid_obj.merge(plant_geogrid_from_geo_obj)

        print('*** x0 (updated):', self.plant_geogrid_obj.x0)
        print('*** xf (updated):', self.plant_geogrid_obj.xf)
        print('*** y0 (updated):', self.plant_geogrid_obj.y0)
        print('*** yf (updated):', self.plant_geogrid_obj.yf)
        print('*** step_x (updated):',
              self.plant_geogrid_obj.step_x)
        print('*** step_y (updated):',
              self.plant_geogrid_obj.step_y)

        if flag_update_geo:
            geo.geogrid(self.plant_geogrid_obj.x0,
                        self.plant_geogrid_obj.y0,
                        self.plant_geogrid_obj.step_x,
                        self.plant_geogrid_obj.step_y,
                        self.plant_geogrid_obj.width,
                        self.plant_geogrid_obj.length,
                        self.epsg)

    def get_coordinates_from_h5_file(self, nisar_product_obj):

        polygon = nisar_product_obj.identification.boundingPolygon
        print('bounding polygon:')
        with plant.PlantIndent():
            bounds = shapely.wkt.loads(polygon).bounds

            yf = bounds[1]
            y0 = bounds[3]
            xf = bounds[2]
            x0 = bounds[0]
            print('polygon WKT:', polygon)
            print('bounding box:')
            with plant.PlantIndent():
                print('min lat:', yf)
                print('min lon:', x0)
                print('max lat:', y0)
                print('max lon:', xf)
                if self.epsg is None:
                    zones_list = []
                    for lat in [y0, yf]:
                        for lon in [x0, xf]:
                            zones_list.append(plant_isce3.point2epsg(lon, lat))
                    vals, counts = np.unique(zones_list, return_counts=True)
                    self.epsg = int(vals[np.argmax(counts)])
                    print('closest projection EPSG code supported by NISAR:',
                          self.epsg)

                y0, x0 = self.proj_inverse(y0, x0, self.epsg)

                yf, xf = self.proj_inverse(yf, xf, self.epsg)

                print('min y:', yf)
                print('min y:', x0)
                print('max x:', y0)
                print('max x:', xf)

                bbox = plant.get_bbox(x0=x0, xf=xf, y0=y0, yf=yf)
                projection = plant.epsg_to_wkt(self.epsg)
                rslc_geogrid_obj = plant.get_coordinates(bbox=bbox,
                                                         projection=projection)
                self.plant_geogrid_obj.merge(rslc_geogrid_obj)

    def update_geogrid_from_isce3_geogrid(self, isce3_geogrid, geo=None):

        flag_update_geo = geo is not None

        width = self.plant_geogrid_obj.width
        length = self.plant_geogrid_obj.length
        x0_orig = self.plant_geogrid_obj.x0
        y0_orig = self.plant_geogrid_obj.y0
        step_x = self.plant_geogrid_obj.step_x
        step_y = self.plant_geogrid_obj.step_y

        if width is None:
            width = -9999
        if length is None:
            length = -9999

        if x0_orig is None:
            x0_orig = np.nan
        if y0_orig is None:
            y0_orig = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_x):
            step_x = plant.m_to_deg_lon(30.)
        elif step_x is None:
            step_x = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_y):
            step_y = - plant.m_to_deg_lat(30.)
        elif step_y is None:
            step_y = np.nan

        print('*** x0:', x0_orig)
        print('*** y0:', y0_orig)
        print('*** step_x:', step_x)
        print('*** step_y:', step_y)
        print('*** length:', length)
        print('*** width:', width)
        projection = plant.epsg_to_wkt(isce3_geogrid.epsg)

        plant_geogrid_from_geo_obj = plant.PlantGeogrid(
            y0=isce3_geogrid.start_y,
            length=isce3_geogrid.length,
            x0=isce3_geogrid.start_x,
            width=isce3_geogrid.width,
            step_x=isce3_geogrid.spacing_x,
            step_y=isce3_geogrid.spacing_y,
            projection=projection)

        print('*** x0 (from geo 2):', plant_geogrid_from_geo_obj.x0)
        print('*** xf (from geo 2):', plant_geogrid_from_geo_obj.xf)
        print('*** y0 (from geo 2):', plant_geogrid_from_geo_obj.y0)
        print('*** yf (from geo 2):', plant_geogrid_from_geo_obj.yf)
        print('*** step_x (from geo 2):', plant_geogrid_from_geo_obj.step_x)
        print('*** step_y (from geo 2):', plant_geogrid_from_geo_obj.step_y)

        self.plant_geogrid_obj.merge(plant_geogrid_from_geo_obj)

        print('*** x0 (updated):', self.plant_geogrid_obj.x0)
        print('*** xf (updated):', self.plant_geogrid_obj.xf)
        print('*** y0 (updated):', self.plant_geogrid_obj.y0)
        print('*** yf (updated):', self.plant_geogrid_obj.yf)
        print('*** step_x (updated):',
              self.plant_geogrid_obj.step_x)
        print('*** step_y (updated):',
              self.plant_geogrid_obj.step_y)

        if flag_update_geo:
            geo.geogrid(self.plant_geogrid_obj.x0,
                        self.plant_geogrid_obj.y0,
                        self.plant_geogrid_obj.step_x,
                        self.plant_geogrid_obj.step_y,
                        self.plant_geogrid_obj.width,
                        self.plant_geogrid_obj.length,
                        self.epsg)

    def proj_inverse(self, lat, lon, epsg):
        if epsg is None:
            return lat, lon
        wgs84_coordinate_system = osr.SpatialReference()
        wgs84_coordinate_system.SetWellKnownGeogCS("WGS84")
        try:
            wgs84_coordinate_system.SetAxisMappingStrategy(
                osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            pass

        projected_coordinate_system = osr.SpatialReference()
        projected_coordinate_system.ImportFromEPSG(epsg)
        try:
            projected_coordinate_system.SetAxisMappingStrategy(
                osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            pass

        transformation = osr.CoordinateTransformation(
            wgs84_coordinate_system, projected_coordinate_system)
        x_out, y_out, _ = transformation.TransformPoint(lon, lat, 0)
        return y_out, x_out

    def get_isce3_geogrid(self, *args, **kwargs):
        self.update_geogrid(*args, **kwargs)

        geogrid = isce3.product.GeoGridParameters(
            start_x=self.plant_geogrid_obj.x0,
            start_y=self.plant_geogrid_obj.y0,
            spacing_x=self.plant_geogrid_obj.step_x,
            spacing_y=self.plant_geogrid_obj.step_y,
            width=self.plant_geogrid_obj.width,
            length=self.plant_geogrid_obj.length,
            epsg=self.epsg)

        return geogrid

    def _create_output_raster(self, filename, nbands=1,
                              gdal_dtype=gdal.GDT_Float32,
                              width=None, length=None):
        if not filename:
            return

        if width is None:
            width = self.plant_geogrid_obj.width
        if length is None:
            length = self.plant_geogrid_obj.length

        output_dir = os.path.dirname(filename)

        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_format = plant_isce3.get_isce3_temporary_format(filename)

        print(f'creating file: {filename}')

        with plant.PlantIndent():
            print(f'length: {length}')
            print(f'width: {width}')
            print(f'nbands: {nbands}')
            print(f'output format: {output_format}')
            print(f'GDAL data type: {gdal_dtype}')

        output_obj = plant_isce3.get_isce3_raster(
            filename,
            int(width),
            int(length),
            int(nbands),
            int(gdal_dtype),
            output_format)

        return output_obj

    def _symmetrize_cross_pols(self, hv_ref, vh_ref):
        print(f'Symmetrizing: {hv_ref} and {vh_ref}')
        hv_raster_obj = plant_isce3.get_isce3_raster(hv_ref)
        vh_raster_obj = plant_isce3.get_isce3_raster(vh_ref)
        width = hv_raster_obj.width
        length = hv_raster_obj.length
        gdal_dtype = hv_raster_obj.datatype()
        temp_symmetrized_file = plant.get_temporary_file(
            append=True, suffix='_symmetrized', ext='tif')
        print('*** temporary symmetrized file:'
              f' {temp_symmetrized_file}')
        symmetrized_hv_raster_obj = self._create_output_raster(
            temp_symmetrized_file, nbands=1, gdal_dtype=gdal_dtype,
            width=width, length=length)
        isce3.polsar.symmetrize_cross_pol_channels(
            hv_raster_obj, vh_raster_obj, symmetrized_hv_raster_obj)
        del symmetrized_hv_raster_obj
        return temp_symmetrized_file

    def _get_symmetrized_input_raster(self, image_obj, temp_file,
                                      temp_symmetrized_file,
                                      hv_band=None, vh_band=None,
                                      output_format=None):
        flag_symmetrize = getattr(self, 'flag_symmetrize', None)

        with plant.PlantIndent():
            output_band = 0
            for b in range(image_obj.nbands):
                band = image_obj.get_band(band=b)
                if (flag_symmetrize and
                        ((vh_band is not None and vh_band == b) or
                         (band.name is not None and
                          band.name.upper() == 'VH'))):
                    print('*** skipping VH')
                    continue
                if (flag_symmetrize and
                        ((hv_band is not None and hv_band == b) or
                         (band.name is not None and
                          band.name.upper() == 'HV'))):
                    symmetrized_hv_obj = self.read_image(temp_symmetrized_file)
                    symmetrized_band = symmetrized_hv_obj.band
                    image_obj.set_band(symmetrized_band, band=output_band)

                    output_band += 1
                    print('*** skipping HV')
                    print('*** reading symmetrized HV')
                    continue
                print(f'*** adding {band} to VRT file')
                image_obj.set_band(band, band=output_band)
                output_band += 1
            if flag_symmetrize:
                image_obj.set_nbands(image_obj.nbands - 1,
                                     realize_changes=False)
            self.save_image(image_obj, temp_file, force=True,
                            output_format=output_format)

    def get_input_raster_from_nisar_product(self, *args, **kwargs):

        with plant.PlantIndent():
            input_raster = self._get_input_raster_from_nisar_product(*args,
                                                                     **kwargs)

        return input_raster

    def get_nlooks(self, frequency=None):

        if frequency is None:
            frequency = self.get_frequency()

        if frequency is not None:
            frequency_lower = frequency.lower()

        if (frequency is not None and
                self.getattr2(f'nlooks_y_{frequency_lower}') is not None):
            nlooks_y = self.getattr2(f'nlooks_y_{frequency_lower}')
        elif self.getattr2('nlooks_y') is not None:
            nlooks_y = self.getattr2('nlooks_y')
        else:
            nlooks_y = 1

        if (frequency is not None and
                self.getattr2(f'nlooks_x_{frequency_lower}') is not None):
            nlooks_x = self.getattr2(f'nlooks_x_{frequency_lower}')
        elif self.getattr2('nlooks_x') is not None:
            nlooks_x = self.getattr2('nlooks_x')
        else:
            nlooks_x = 1

        return nlooks_y, nlooks_x

    def _get_input_raster_from_nisar_product(self, input_raster=None,
                                             input_file=None,
                                             plant_product_obj=None,
                                             frequency=None):

        if (input_raster is None and
                getattr(self, 'input_raster', None) is not None):
            input_raster = self.input_raster

        if input_file is None:
            input_file = self.input_file

        flag_transform_input_raster = \
            getattr(self, 'flag_transform_input_raster', None)
        flag_symmetrize = \
            getattr(self, 'flag_symmetrize', None)

        symmetrize_bands = \
            getattr(self, 'symmetrize_bands', None)

        if frequency is None and plant_product_obj is None:
            print('ERROR frequency and plant_product_obj cannot both be'
                  ' None in the call to get_input_raster_from_nisar_product()')
            return

        if frequency is None:
            frequency = plant_product_obj.get_frequency()

        if input_raster is not None:

            if flag_transform_input_raster is not False:

                flag_apply_transformation = \
                    self.plant_transform_obj.flag_apply_transformation()
                image_obj = self.read_image(input_raster)

            else:
                flag_apply_transformation = False
                image_obj = plant.read_image(input_raster)

            self.print('*** flag_apply_transformation:'
                       f' {flag_apply_transformation}')
            self.print(f'*** transformation: {self.plant_transform_obj}')
            if flag_apply_transformation:

                temp_file = plant.get_temporary_file(append=True,
                                                     ext='tif')
                self.print(f'*** creating temporary file (1): {temp_file}')
                self.save_image(image_obj, temp_file, force=True)
                input_raster = temp_file

            if flag_symmetrize and symmetrize_bands is None:
                self.print('ERROR symmetrization option with input raster'
                           ' requires the parameter --symmetrize-bands')
                return
            elif flag_symmetrize:
                self.print('applying polarimetric symmetrization to input'
                           ' raster')
                hv_band = symmetrize_bands[0]
                vh_band = symmetrize_bands[1]

                hv_obj = plant.read_image(input_raster, band=hv_band)
                temp_hv_file = plant.get_temporary_file(
                    append=True, suffix='_hv', ext='vrt')
                plant.save_image(hv_obj, temp_hv_file, force=True,
                                 output_format='VRT')

                vh_obj = plant.read_image(input_raster, band=vh_band)
                temp_vh_file = plant.get_temporary_file(
                    append=True, suffix='_vh', ext='vrt')
                plant.save_image(vh_obj, temp_vh_file, force=True,
                                 output_format='VRT')

                temp_symmetrized_file = self._symmetrize_cross_pols(
                    temp_hv_file, temp_vh_file)

                temp_file = plant.get_temporary_file(
                    append=True, suffix='_input_raster_symmetrized', ext='tif')
                image_obj = self.read_image(input_raster)

                self._get_symmetrized_input_raster(
                    image_obj, temp_file, temp_symmetrized_file,
                    hv_band=hv_band, vh_band=vh_band,
                    output_format='TIFF')
                input_raster = temp_file

        else:
            self.print(f'selecting product frequency: {frequency}')

            nisar_product_obj = open_product(input_file)

            if nisar_product_obj.getProductLevel() == "L1":
                imagery_path = (f'{nisar_product_obj.SwathPath}/'
                                f'frequency{frequency}')
            else:
                imagery_path = (f'{nisar_product_obj.GridPath}/'
                                f'frequency{frequency}')

            if flag_symmetrize:
                hv_ref = f'HDF5:{input_file}:{imagery_path}/HV'
                vh_ref = f'HDF5:{input_file}:{imagery_path}/VH'
                temp_symmetrized_file = self._symmetrize_cross_pols(
                    hv_ref, vh_ref)

            else:
                temp_symmetrized_file = None

            raster_file = f'NISAR:{input_file}:{frequency}'
            temp_file = plant.get_temporary_file(append=True,
                                                 ext='vrt')
            self.print(f'*** creating temporary file (2): {temp_file}')
            image_obj = self.read_image(raster_file)

            self._get_symmetrized_input_raster(
                image_obj, temp_file, temp_symmetrized_file,
                output_format='VRT')
            input_raster = temp_file

        nlooks_y, nlooks_x = self.get_nlooks(frequency)

        if nlooks_y > 1 or nlooks_x > 1:

            image_obj = plant.read_image(input_raster)

            self.print('multilooking input file')
            dtype_str = plant.get_dtype_name(image_obj.dtype)

            filter_kwargs = {}

            self.print(f'data type: {dtype_str}')
            if ('COMPLEX' in dtype_str.upper() or
                    'CFLOAT' in dtype_str.upper()):
                exponent = 2

                filter_kwargs['transform_square'] = True
            else:
                exponent = 1

            self.print(f'exponent: {exponent}')
            self.print(f'number of looks: {nlooks_y} (az) x'
                       f' {nlooks_x} (rg)')
            self.print(f'original: {image_obj.length} (length) x'
                       f' {image_obj.width} (width)')

            temp_file = plant.get_temporary_file(append=True,
                                                 ext='tif')

            plant_isce3.multilook_isce3(input_raster,
                                        output_file=temp_file,
                                        nlooks_y=nlooks_y,
                                        nlooks_x=nlooks_x,

                                        **filter_kwargs)

            image_obj = plant.read_image(temp_file)

            self.print(f'multilooked: {image_obj.length} (length) x'
                       f' {image_obj.width} (width)')

            input_raster = temp_file

        return input_raster

    def get_frequency(self):

        frequency = self.getattr2('frequency')

        return frequency

    def get_radar_grid_ml(self, radar_grid, frequency=None):

        if (self.getattr2('select_row') is not None or
                self.getattr2('select_col') is not None):
            self.plant_transform_obj.update_crop_window(
                length_orig=radar_grid.length,
                width_orig=radar_grid.width)
            y0 = self.plant_transform_obj._offset_y
            if y0 is None:
                y0 = 0
            x0 = self.plant_transform_obj._offset_x
            if x0 is None:
                x0 = 0
            length = self.plant_transform_obj.length
            if length is None:
                length = radar_grid.length
            width = self.plant_transform_obj.width
            if width is None:
                width = radar_grid.width
            radar_grid = radar_grid.offset_and_resize(
                y0, x0, length, width)

        nlooks_y, nlooks_x = self.get_nlooks(frequency=frequency)

        if nlooks_y > 1 or nlooks_x > 1:

            self.print('multilooking radar grid')
            radar_grid_ml = radar_grid.multilook(nlooks_y, nlooks_x)
            with plant.PlantIndent():
                self.print(f'number of looks: {nlooks_y} (az) x'
                           f' {nlooks_x} (rg)')
                self.print(f'original: {radar_grid.length} (length) x'
                           f' {radar_grid.width} (width)')
                self.print(f'multilooked: {radar_grid_ml.length} (length) x'
                           f' {radar_grid_ml.width} (width)')
        else:
            radar_grid_ml = radar_grid

        return radar_grid_ml

    def get_binary_water_mask_ctable(self):
        mask_ctable = gdal.ColorTable()

        mask_ctable.SetColorEntry(0, (0, 0, 255))

        mask_ctable.SetColorEntry(1, (255, 255, 255))

        mask_ctable.SetColorEntry(255, (0, 0, 0))

        return mask_ctable

    def get_mask_ctable(self, mask_array):
        mask_ctable = gdal.ColorTable()

        mask_ctable.SetColorEntry(0, (175, 175, 175))

        mask_ctable.SetColorEntry(255, (0, 0, 0))

        if not self.cmap:
            self.cmap = 'viridis'

        n_subswaths = min(np.max(mask_array[(mask_array < 10)]), 5)
        print('number of subswaths:', n_subswaths)

        for subswath in range(1, n_subswaths + 1):
            color = plant.get_color_display(subswath + 1,
                                            flag_decreasing=True,
                                            n_colors=n_subswaths + 2,
                                            cmap=self.cmap)
            color_rgb = tuple([int(255 * x) for x in color[0:3]])
            mask_ctable.SetColorEntry(subswath, color_rgb)
        return mask_ctable

    def get_layover_shadow_mask_ctable(self):
        layover_shadow_mask_ctable = gdal.ColorTable()

        layover_shadow_mask_ctable.SetColorEntry(0, (175, 175, 175))

        layover_shadow_mask_ctable.SetColorEntry(1, (64, 64, 64))

        layover_shadow_mask_ctable.SetColorEntry(2, (223, 223, 223))

        layover_shadow_mask_ctable.SetColorEntry(3, (0, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(11, (32, 32, 32))

        layover_shadow_mask_ctable.SetColorEntry(13, (0, 128, 128))

        layover_shadow_mask_ctable.SetColorEntry(22, (255, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(23, (128, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(33, (128, 128, 128))

        layover_shadow_mask_ctable.SetColorEntry(255, (0, 0, 0))
        return layover_shadow_mask_ctable

    def get_dem_interp_method(self):
        return self.get_interp_method(self.dem_interp_method)

    def get_data_interp_method(self):
        return self.get_interp_method(self.data_interp_method)

    def get_interp_method(self, interp_method):

        if (interp_method is not None and
                interp_method.upper() == 'SINC'):
            interp_method_obj = isce3.core.DataInterpMethod.SINC
        elif (interp_method is not None and
                interp_method.upper() == 'BILINEAR'):
            interp_method_obj = isce3.core.DataInterpMethod.BILINEAR
        elif (interp_method is not None and
                interp_method.upper() == 'BICUBIC'):
            interp_method_obj = isce3.core.DataInterpMethod.BICUBIC
        elif (interp_method is not None and
                interp_method.upper() == 'NEAREST'):
            interp_method_obj = isce3.core.DataInterpMethod.NEAREST
        else:
            interp_method_obj = isce3.core.DataInterpMethod.BIQUINTIC

        return interp_method_obj

    def get_doppler_grid_lut_nisar(self, nisar_product_obj):

        if self.native_doppler_grid:
            print('*** Grid: native dop')
            doppler = nisar_product_obj.getDopplerCentroid()
            doppler.bounds_error = False
        else:

            print('*** Grid: zero dop')
            doppler = isce3.core.LUT2d()
        return doppler

    def get_doppler_grid_lut_s1(self, burst):

        if self.native_doppler_grid:
            print('*** Grid: native dop (Sentinel-1)')
            doppler = burst.doppler.lut2d
            doppler.bounds_error = False
        else:

            print('*** Grid: zero dop (Sentinel-1)')
            doppler = isce3.core.LUT2d()
        return doppler

    def get_doppler_centroid_lut_nisar(self, nisar_product_obj):

        if self.zero_doppler_centroid:

            print('*** Doppler Centroid: zero dop (NISAR)')
            doppler_centroid_lut = isce3.core.LUT2d()
        else:
            print('*** Doppler Centroid: native dop (NISAR)')
            doppler_centroid_lut = nisar_product_obj.getDopplerCentroid()
            doppler_centroid_lut.bounds_error = False
        return doppler_centroid_lut

    def get_doppler_centroid_lut_s1(self, burst):

        if self.zero_doppler_centroid:

            print('*** Doppler Centroid: native dop (Sentinel-1)')
            doppler_centroid_lut = burst.doppler.lut2d
            doppler_centroid_lut.bounds_error = False
        else:

            print('*** Doppler Centroid: zero dop (Sentinel-1)')
            doppler_centroid_lut = isce3.core.LUT2d()
        return doppler_centroid_lut

    def parse_nisar_s3_path(self, s3_path):
        s3_path_splitted = s3_path.split('/')

        year_range = [str(y) for y in range(2025, 2050)]
        year_offset = None
        for i, s3_path_part in enumerate(s3_path_splitted):
            if s3_path_part not in year_range:
                continue
            year_offset = i
            break
        if year_offset is None:
            raise ValueError(f'Could not find year in S3 path: {s3_path}')
        ret_dict = {}
        ret_dict['year'] = s3_path_splitted[year_offset]
        ret_dict['month'] = s3_path_splitted[year_offset + 1]
        ret_dict['day'] = s3_path_splitted[year_offset + 2]

        return ret_dict

    def parse_nisar_product_filename(self, filename_with_extension):
        return parse_nisar_product_filename(filename_with_extension)

    def get_masked_nisar_data_radar_coordinates(
            self, nisar_product_obj, image,
            freq, nlooks_y, nlooks_x):
        try:
            swaths_base_path = nisar_product_obj.SwathPath
        except BaseException:
            print('ERROR could not get swath path'
                  ' from provided product. Ensure the product'
                  ' is a level 1 product.')
            return

        masked_image = np.full_like(image, np.nan)

        with plant.h5py_file_wrapper(self.input_file, 'r') as root_ds:
            for i in range(1, 6):
                valid_samples_path = (
                    f'{swaths_base_path}/frequency{freq}/'
                    f'validSamplesSubSwath{i}')

                if valid_samples_path not in root_ds:
                    continue

                valid_samples_array = root_ds[valid_samples_path][()]

                for row in range(image.shape[0]):

                    start = np.nanmedian(valid_samples_array[
                        row * nlooks_y: (row + 1) * nlooks_y, 0], axis=0)
                    stop = np.nanmedian(valid_samples_array[
                        row * nlooks_y: (row + 1) * nlooks_y, 1], axis=0)

                    if start < 0 or stop < 0:
                        continue

                    start_ml = max(int(np.ceil(start / nlooks_x)), 0)
                    stop_ml = min(int(np.floor(stop / nlooks_x)) + 1,
                                  image.shape[1])

                    masked_image[row, start_ml: stop_ml] = \
                        image[row, start_ml: stop_ml]

        return masked_image

    def compute_binned_aggregation(self, x_image, y_image,
                                   x_min, x_max, x_step):

        if x_image.shape != y_image.shape:
            raise ValueError(f"x_image shape {x_image.shape}"
                             f" != y_image shape {y_image.shape}")

        x = x_image.ravel()
        y = y_image.ravel()

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        bins = np.arange(x_min, x_max + x_step, x_step)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_idx = np.digitize(x, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(bin_centers))

        bin_idx = bin_idx[valid]
        y = y[valid]

        def compute_means(values):
            sums = np.bincount(bin_idx, weights=values,
                               minlength=len(bin_centers))
            counts = np.bincount(bin_idx, minlength=len(bin_centers))
            with np.errstate(invalid='ignore', divide='ignore'):
                means = sums / counts
            means[counts == 0] = np.nan
            return means

        if np.iscomplexobj(y):
            means_real = compute_means(np.real(y))
            means_imag = compute_means(np.imag(y))
            means = means_real + 1j * means_imag
        else:
            means = compute_means(y)

        return bin_centers, means

    def generate_elevation_profiles(
            self, output_dir,
            nisar_product_obj, freq, pol,
            image,
            radar_grid_ml,
            metadata_dict,
            profile_filter_size,

            suffix_ml,
            prefix='',
            flag_phase=False):

        width = image.shape[1]

        output_elevation_image = os.path.join(
            output_dir, 'elevation',
            f'elevation_image_{freq}{suffix_ml}.tif')

        if os.path.isfile(output_elevation_image):

            print("elevation image file already exists, reading from file:",
                  output_elevation_image)
            elevation_image = plant.read_image(output_elevation_image).image

        else:

            print('generating elevation profiles for frequency', freq)
            dem_raster = isce3.io.Raster(self.dem_file)

            orbit = nisar_product_obj.getOrbit()

            native_doppler = nisar_product_obj.getDopplerCentroid()
            native_doppler.bounds_error = False

            grid_doppler = isce3.core.LUT2d()
            epsg = 4326
            dem_interp_method = isce3.core.DataInterpMethod.BIQUINTIC
            rdr2geo_params = isce3.geometry.Rdr2GeoParams()
            geo2rdr_params = isce3.geometry.Geo2RdrParams()

            interpolated_dem_raster = None
            coordinate_x_raster = None
            coordinate_y_raster = None
            incidence_angle_raster = None
            los_unit_vector_x_raster = None
            los_unit_vector_y_raster = None
            along_track_unit_vector_x_raster = None
            along_track_unit_vector_y_raster = None

            ground_track_velocity_raster = None

            nbands = 1
            length = radar_grid_ml.length
            width = radar_grid_ml.width

            print('*** elevation image length:', length)
            print('*** elevation image width:', width)

            output_elevation_image_dir = os.path.dirname(
                output_elevation_image)
            os.makedirs(output_elevation_image_dir, exist_ok=True)
            elevation_angle_raster = isce3.io.Raster(
                output_elevation_image,

                width,
                length,
                nbands,
                gdal.GDT_Float32,
                "GTiff")

            isce3.geometry.get_geolocation_grid(
                dem_raster,
                radar_grid_ml,
                orbit,
                native_doppler,
                grid_doppler,
                epsg,
                dem_interp_method,
                rdr2geo_params,
                geo2rdr_params,
                interpolated_dem_raster,
                coordinate_x_raster,
                coordinate_y_raster,
                incidence_angle_raster,
                los_unit_vector_x_raster,
                los_unit_vector_y_raster,
                along_track_unit_vector_x_raster,
                along_track_unit_vector_y_raster,
                elevation_angle_raster,
                ground_track_velocity_raster)

            elevation_angle_raster.close_dataset()

            if os.path.isfile(output_elevation_image):
                print("## file saved:", output_elevation_image)
                self.output_files.append(output_elevation_image)

            elevation_image = \
                plant.read_image(output_elevation_image).image

            elevation_image_max = np.nanmax(elevation_image)
            if (not np.isfinite(elevation_image_max) or
                    elevation_image_max == 0):
                print('ERROR elevation image max value is not finite or is 0,'
                      f' check elevation image for frequency {freq}:'
                      f' {output_elevation_image}')
                return

        elevation_min = 30.0
        elevation_max = 41.5
        elevation_step = (elevation_max - elevation_min) / 100.0

        print('computing elevation profiles for frequency', freq)
        print('    elevation min:', elevation_min)
        print('    elevation max:', elevation_max)
        print('    elevation step:', elevation_step)

        elevation_profile, image_profile = \
            self.compute_binned_aggregation(
                elevation_image, image,
                elevation_min, elevation_max, elevation_step)

        print('elevation profile shape:', elevation_profile.shape)
        print('image profile shape:', image_profile.shape)

        if profile_filter_size > 0:
            print('applying profile filter of size',
                  profile_filter_size)

            image_profile = plant.filter_data(
                image_profile,
                mean=[1, profile_filter_size])

        elevation_profile_file = os.path.join(
            output_dir, 'elevation',
            f'elevation_profile_{freq}.tif')

        plant.save_image(elevation_profile,
                         output_file=elevation_profile_file,
                         metadata=metadata_dict,
                         force=True)

        image_median = np.nanmedian(image_profile)
        print(f'image mean ({pol}): {image_median}')
        if not np.isfinite(image_median):
            image_profile_file = os.path.join(
                output_dir, 'elevation',
                f'{prefix}profile_{freq}_{pol}_with_offset_error.tif')
            plant.save_image(image_profile,
                             output_file=image_profile_file,
                             metadata=metadata_dict,
                             force=True)
            print(f'ERROR image mean for frequency {freq},'
                  f' polarization {pol} is not finite: {image_median}')
            return

        if flag_phase:
            image_profile = np.angle(
                image_profile * np.exp(-1j * np.angle(image_median)))
        else:
            image_profile /= image_median

        image_profile_file = os.path.join(
            output_dir, 'elevation',
            f'{prefix}profile_{freq}_{pol}.tif')

        plant.save_image(image_profile,
                         output_file=image_profile_file,
                         metadata=metadata_dict,
                         force=True)

        return elevation_profile, image_profile, image_median


def parse_nisar_product_filename(filename_with_extension):
    filename = os.path.basename(filename_with_extension).split('.')[0]
    splitted_filename = filename.split('_')
    ret_dict = {}

    if splitted_filename[0] != 'NISAR':
        raise ValueError(f'Unrecognized filename format: {filename}')

    instrument_char = splitted_filename[1][0]
    if instrument_char == 'L':
        instrument = 'LSAR'
    elif instrument_char == 'S':
        instrument = 'SSAR'
    else:
        raise ValueError(
            f'Unrecognized instrument character "{instrument_char}"'
            ' in filename: {filename}')
    ret_dict['instrument_char'] = instrument_char
    ret_dict['instrument'] = instrument

    level = int(splitted_filename[1][1])
    ret_dict['level'] = level

    processing_type_chars = splitted_filename[2]
    if processing_type_chars == 'PR':
        processing_type = 'Production'
    elif processing_type_chars == 'UR':
        processing_type = 'Urgent Response'
    else:
        raise ValueError(
            'Unrecognized processing type characters'
            f' "{processing_type_chars}"'
            f' in filename: {filename}')

    ret_dict['processing_type_chars'] = processing_type_chars
    ret_dict['processing_type'] = processing_type

    product_type = splitted_filename[3]
    ret_dict['product_type'] = product_type

    if product_type not in ['RRSD', 'RSLC', 'GCOV', 'GSLC', 'SME2']:
        raise ValueError(
            'Not supported product type "{product_type}"'
            f' in filename: {filename}')

    cycle_number = int(splitted_filename[4])
    ret_dict['cycle_number'] = cycle_number

    track_number = int(splitted_filename[5])
    ret_dict['track_number'] = track_number

    orbit_direction_char = splitted_filename[6]
    if orbit_direction_char == 'A':
        orbit_direction = 'Ascending'
    elif orbit_direction_char == 'D':
        orbit_direction = 'Descending'
    else:
        raise ValueError(
            'Unrecognized orbit direction character'
            f' "{orbit_direction_char}"'
            f' in filename: {filename}')
    ret_dict['orbit_direction_char'] = orbit_direction_char
    ret_dict['orbit_direction'] = orbit_direction

    if product_type == 'RRSD':
        radar_configuration_mode = splitted_filename[7][0:3]
        ret_dict['radar_configuration_mode'] = radar_configuration_mode
        radar_processing_mode = splitted_filename[7][3]
        ret_dict['radar_processing_mode'] = radar_processing_mode
        offset = 8
    else:
        frame_number = int(splitted_filename[7])
        ret_dict['frame_number'] = frame_number
        mode_freq_a = splitted_filename[8][0:2]
        mode_freq_b = splitted_filename[8][2:4]
        ret_dict['mode_freq_a'] = mode_freq_a
        ret_dict['mode_freq_b'] = mode_freq_b
        pol_freq_a = splitted_filename[9][0:2]
        pol_freq_b = splitted_filename[9][2:4]
        ret_dict['pol_freq_a'] = pol_freq_a
        ret_dict['pol_freq_b'] = pol_freq_b
        radar_processing_mode = splitted_filename[10]
        ret_dict['radar_processing_mode'] = radar_processing_mode
        offset = 11

    start_date_time = splitted_filename[offset]
    ret_dict['start_date_time'] = start_date_time
    end_date_time = splitted_filename[offset + 1]
    ret_dict['end_date_time'] = end_date_time
    crid = splitted_filename[offset + 2]
    ret_dict['crid'] = crid
    accuracy = splitted_filename[offset + 3]
    ret_dict['accuracy'] = accuracy

    if product_type != 'RRSD':
        coverage_char = splitted_filename[offset + 4]
        if coverage_char == 'F':
            coverage = 'Full'
        elif coverage_char == 'P':
            coverage = 'Partial'
        else:
            raise ValueError(
                'Unrecognized coverage character'
                f' "{coverage_char}" in filename: {filename}')
        ret_dict['coverage_char'] = coverage_char
        ret_dict['coverage'] = coverage
        offset_loc_char = offset + 5
    else:
        offset_loc_char = offset + 4

    location_char = splitted_filename[offset_loc_char]
    if location_char == 'J':
        location = 'JPL'
    elif location_char == 'N':
        location = 'NRSC'
    else:
        raise ValueError(
            'Unrecognized location character'
            f' "{location_char}" in filename: {filename}')
    ret_dict['location_char'] = location_char
    ret_dict['location'] = location

    counter = splitted_filename[offset_loc_char + 1]
    ret_dict['counter'] = counter

    granule_id = '_'.join(splitted_filename[0: offset_loc_char + 2])
    ret_dict['granule_id'] = granule_id

    if len(splitted_filename) > offset_loc_char + 2:
        for i in range(offset_loc_char + 2, len(splitted_filename)):
            if splitted_filename[i] in ['HH', 'HV', 'VH', 'VV']:
                ret_dict['polarization'] = splitted_filename[i]
            if splitted_filename[i] in ['A', 'B']:
                ret_dict['frequency'] = splitted_filename[i]

    return ret_dict


def get_raster_from_data(data, scratch_path='.'):
    temp_file = plant.get_temporary_file(ext='.bin', append=True)

    length, width = data.shape

    driver = gdal.GetDriverByName("ENVI")

    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(
        data.dtype)

    dset = driver.Create(temp_file, xsize=width, ysize=length,
                         bands=1, eType=dtype)
    raster_band = dset.GetRasterBand(1)
    raster_band.WriteArray(data)

    raster_band.FlushCache()
    del raster_band
    del dset
    gc.collect()

    temp_raster = plant_isce3.get_isce3_raster(temp_file)
    return temp_raster


def get_raster_from_lut(correction_lut, radar_grid):
    data = correction_lut.data

    correction_raster = get_raster_from_data(data)

    radar_grid = isce3.product.RadarGridParameters(
        correction_lut.y_start,
        radar_grid.wavelength,
        1.0 / correction_lut.y_spacing,
        correction_lut.x_start,
        correction_lut.x_spacing,
        radar_grid.lookside,
        correction_lut.length,
        correction_lut.width,
        radar_grid.ref_epoch)

    return correction_raster, radar_grid


def _get_output_dict_from_parser(parser, args, module_name):
    orig_index = []
    if isinstance(args, dict):
        args_keys = args.keys()
        kwargs = args
    else:
        args_keys = []
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                args_keys.append(arg[2:])
                orig_index.append(i)
            elif arg.startswith('-') and not plant.isnumeric(arg[1:]):
                args_keys.append(arg[1:])
                orig_index.append(i)
        args_keys = [x.replace('-', '_').strip('_')
                     for x in args_keys]
        kwargs = None
    ret = plant.get_args_from_argparser(parser,
                                        store_true_action=False,
                                        store_false_action=False,
                                        store_action=True,
                                        help_action=False,
                                        dest='output_file')
    output_file_keys = [x.replace('-', '_').strip('_')
                        for x in ret]
    output_file_keys.append('output_file')

    ret = plant.get_args_from_argparser(parser,
                                        store_true_action=False,
                                        store_false_action=False,
                                        store_action=True,
                                        help_action=False,
                                        dest='output_dir')
    output_dir_keys = [x.replace('-', '_').strip('_')
                       for x in ret]
    output_dir_keys.append('output_dir')

    ret = plant.get_args_from_argparser(parser,
                                        dest='output_file')
    flag_output = bool(ret)
    output_key = None

    for key in output_file_keys:
        if key not in args_keys:
            continue
        output_key = key
        if isinstance(args, dict):
            break
        output_key_index = orig_index[args_keys.index(output_key)]
        break

    output_str = ''
    output_args = []
    flag_new_mem_output = False

    if flag_output and output_key:
        if kwargs is not None:
            value_str = kwargs[output_key]
        else:
            value_str = args[output_key_index + 1]

        output_str = f' {ret[0]} {value_str}'
        output_args.append(ret[0])
        output_args.append(value_str)

    elif (flag_output and
          not any([key in args_keys
                   for key in output_dir_keys]) and
          module_name != 'plant_display'):

        mem_output_str = 'MEM:' + plant.get_temporary_file()
        output_str = f' {ret[0]} {mem_output_str}'
        output_args.append(ret[0])
        output_args.append(mem_output_str)
        flag_new_mem_output = True

    output_dict = {}
    output_dict['output_str'] = output_str
    output_dict['output_args'] = output_args
    output_dict['output_file_keys'] = output_file_keys
    output_dict['output_dir_keys'] = output_dir_keys
    output_dict['output_file_keys'] = output_file_keys
    output_dict['flag_output'] = flag_output
    output_dict['flag_new_mem_output'] = flag_new_mem_output

    return output_dict


def compute_correction_lut(burst_in, dem_raster,
                           rg_step_meters,
                           az_step_meters,
                           apply_bistatic_delay_correction,
                           apply_static_tropospheric_delay_correction):

    rg_lut = None
    az_lut = None

    if (not apply_bistatic_delay_correction and
            not apply_static_tropospheric_delay_correction):
        return rg_lut, az_lut

    numrow_orbit = burst_in.orbit.position.shape[0]
    vel_mid = burst_in.orbit.velocity[numrow_orbit // 2, :]
    spd_mid = np.linalg.norm(vel_mid)
    pos_mid = burst_in.orbit.position[numrow_orbit // 2, :]
    alt_mid = np.linalg.norm(pos_mid)

    r = 6371000.0

    az_step_sec = (az_step_meters * alt_mid) / (spd_mid * r)

    bistatic_delay = burst_in.bistatic_delay(range_step=rg_step_meters,
                                             az_step=az_step_sec)

    if apply_bistatic_delay_correction:
        az_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                                  bistatic_delay.y_start,
                                  bistatic_delay.x_spacing,
                                  bistatic_delay.y_spacing,
                                  -bistatic_delay.data)

    if not apply_static_tropospheric_delay_correction:
        return rg_lut, az_lut

    epsg = dem_raster.get_epsg()
    proj = isce3.core.make_projection(epsg)
    ellipsoid = proj.ellipsoid

    rdr_grid = burst_in.as_isce3_radargrid(az_step=az_step_sec,
                                           rg_step=rg_step_meters)

    grid_doppler = isce3.core.LUT2d()

    rdr2geo_obj = isce3.geometry.Rdr2Geo(rdr_grid, burst_in.orbit,
                                         ellipsoid, grid_doppler,
                                         threshold=1.0e-8)

    height_file = plant.get_temporary_file(ext='tif')
    incidence_angle_file = plant.get_temporary_file(ext='tif')

    topo_output = {height_file: gdal.GDT_Float32,
                   incidence_angle_file: gdal.GDT_Float32}

    raster_list = []
    for fname, dtype in topo_output.items():
        topo_output_raster = isce3.io.Raster(fname,
                                             rdr_grid.width, rdr_grid.length,
                                             1, dtype, 'ENVI')
        raster_list.append(topo_output_raster)

    height_raster, incidence_raster = raster_list

    rdr2geo_obj.topo(dem_raster, x_raster=None, y_raster=None,
                     height_raster=height_raster,
                     incidence_angle_raster=incidence_raster)

    height_raster.close_dataset()
    incidence_raster.close_dataset()

    height_arr =\
        gdal.Open(height_file, gdal.GA_ReadOnly).ReadAsArray()
    incidence_angle_arr =\
        gdal.Open(incidence_angle_file, gdal.GA_ReadOnly).ReadAsArray()

    zenith_path_delay = 2.3
    reference_height = 6000.0
    tropo = (zenith_path_delay
             / np.cos(np.deg2rad(incidence_angle_arr))
             * np.exp(-1 * height_arr / reference_height))

    rg_lut = isce3.core.LUT2d(bistatic_delay.x_start,
                              bistatic_delay.y_start,
                              bistatic_delay.x_spacing,
                              bistatic_delay.y_spacing,
                              tropo)

    return rg_lut, az_lut


def point2epsg(lon, lat):

    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 70.0:
        return 3413
    elif lat <= -70.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(
        'Could not determine projection for {0},{1}'.format(lat, lon))


def execute(command,
            verbose=True,

            return_time=False,
            ignore_exception=False):

    if not isinstance(command, list):
        command_vector = plant.get_command_vector(command)
    else:
        command_vector = command

    if len(command_vector) == 0 and verbose:
        print('WARNING command not identified: ' + command)
        return ['']

    start_time = None
    module_name = command_vector[0]
    argv = command_vector[1:]

    module_name = module_name.replace('.py', '')
    flag_error = False

    module_obj = importlib.import_module('plant_isce3.' + module_name)

    current_script = plant.plant_config.current_script
    method_to_execute = getattr(module_obj, 'main')

    if plant.plant_config.logger_obj is None:
        sink = plant.PlantLogger()
    else:
        sink = plant.PlantIndent()

    with sink:

        if verbose:
            arguments = plant.get_command_line_from_argv(argv)
            command_line = (f'{module_name}.py {arguments}')
            print(f'PLAnT-ISCE3 {plant_isce3.VERSION} (API) -'
                  f' {command_line}')

        parser_ref = plant.argparse()
        ret = plant.get_args_from_argparser(parser_ref,
                                            store_true_action=True,
                                            store_false_action=True,
                                            store_action=False,
                                            help_action=False,
                                            dest='cli_mode')
        has_bash_flag = any([element in argv for element in ret])
        if not has_bash_flag:
            argv.append('--no-bash')

        argparse_method = getattr(module_obj, 'get_parser')
        parser = argparse_method()

        output_dict = _get_output_dict_from_parser(
            parser, argv, module_name)

        flag_output = output_dict['flag_new_mem_output']

        if output_dict['flag_new_mem_output']:
            argv.extend(output_dict['output_args'])
            argv.extend(['-u', '--ul', '10'])

        original_sys_argv = sys.argv
        sys.argv = [module_name + '.py'] + argv
        flag_error = False
        ret = None
        if return_time:
            start_time = time.time()
        try:
            ret = method_to_execute(argv)

        except SystemExit as e:
            if len(e.args) == 0 or e.args[0] != 0:
                flag_error = True
                error_message = plant.get_error_message()
        finally:
            sys.argv = original_sys_argv

        if (flag_error and not ignore_exception and
                error_message and 'ERROR' in error_message):
            print(error_message)
        elif flag_error and not ignore_exception:
            print('ERROR executing PLAnT module %s: %s.'
                  % (module_name, error_message))
        if return_time:
            ret = time.time() - start_time
        elif flag_output:
            output_ret = plant._get_output_ret_from_plant_config()
            if output_ret is not None:
                ret = output_ret

        if ret is not None:
            ret_str = ('. Returning object class:'
                       f' {ret.__class__.__name__}')
        else:
            ret_str = ''
        if verbose and plant.plant_config.main_script is not None:
            print(f'PLAnT-ISCE3 (API-completed) - {command_line}'
                  f'{ret_str}')

    gc.collect()
    return ret


def get_nisar_dates_by_cycle(cycle_number):
    ascending_node_datetime = \
        datetime.datetime.fromisoformat('20250911T060000')

    start_datetime = \
        (ascending_node_datetime +
         datetime.timedelta(days=12 * (cycle_number - 1)))
    end_datetime = \
        (ascending_node_datetime +
         datetime.timedelta(days=12 * (cycle_number)))

    return start_datetime, end_datetime


def get_nisar_track_frame_date_from_tfdb(
        track_number, frame_number, track_frame_db_file, cycle_number,
        string_format='%Y/%m/%d'):
    gdf = gpd.read_file(track_frame_db_file)
    start_cy_gdf = gdf[
        (gdf["track"] == track_number) &
        (gdf["frame"] == frame_number)]['startCY']
    start_cy = float(start_cy_gdf)
    ascending_node_datetime = \
        datetime.datetime.fromisoformat('20250911T060000')
    start_datetime = \
        (ascending_node_datetime +
         datetime.timedelta(seconds=start_cy) +
         datetime.timedelta(days=12 * cycle_number))

    if string_format is not None:
        return start_datetime.strftime(string_format)

    return start_datetime


def get_nisar_dates_by_cycle_str_list(cycle_number):
    start_datetime, end_datetime = get_nisar_dates_by_cycle(cycle_number)

    date_strings = [
        (start_datetime + datetime.timedelta(days=i)).strftime("%Y/%m/%d")
        for i in range((end_datetime.date() -
                        start_datetime.date()).days + 1)]

    return date_strings


class ModuleWrapper(object):

    def __init__(self, module_name, *args, ref=None, **kwargs):
        self._module_name = module_name
        self._module_obj = None
        self._ref = ref
        self._args = args
        self._kwargs = kwargs
        self._command = None
        self._set_module_obj(self._module_name)

    def _set_module_obj(self, name):
        self._module_name = self._module_name.replace('.py', '')
        if not self._module_name.startswith('plant_isce3'):
            self._module_name = 'plant_isce3_' + self._module_name
        self._module_obj = importlib.import_module(
            f'plant_isce3.{self._module_name}')

    def __call__(self, *args, **kwargs):
        args = list(self._args) + list(args)
        kwargs = dict(self._kwargs, **kwargs)

        self._set_command(*args, **kwargs)
        if self._command is None:
            return

        flag_mute = kwargs.get('flag_mute', None)
        verbose = kwargs.get('verbose', True) and not flag_mute
        if self._ref is not None:

            ret = self._ref.execute(self._command, verbose=verbose)
        else:

            ret = execute(self._command, verbose=verbose)

        return ret

    def _set_command(self, *args, **kwargs):
        if len(args) > 0:
            args_str = self._update_args_str(
                args, args_str='')
            args_str = ' -i ' + args_str
        else:
            args_str = ''

        argparse_method = getattr(self._module_obj, 'get_parser')
        parser = argparse_method()
        output_dict = _get_output_dict_from_parser(
            parser, kwargs, self._module_name)
        output_file_keys = output_dict['output_file_keys']
        output_str = output_dict['output_str']
        self._flag_output = output_dict['flag_output']

        kwargs_str = ''
        for key, value in kwargs.items():
            if key in output_file_keys:
                continue
            if isinstance(value, list):
                value_str = ''
                for v in value:
                    if value_str:
                        value_str += ' '
                    if isinstance(v, str) and "'" not in v:
                        value_str += "'" + str(v) + "'"
                    elif isinstance(v, str):
                        value_str += '"' + str(v) + '"'
                    else:
                        value_str += str(v)
            elif (isinstance(value, plant.PlantImage) or
                  isinstance(value, np.ndarray)):
                arg_id = str(id(value))
                plant.plant_config.variables[arg_id] = value
                value_str = f' MEM:{arg_id}'
            elif not isinstance(value, str) or '"' not in value:
                value_str = f'"{value}"'
            else:
                value_str = f"'{value}'"

            kwargs_dest = {}
            if key.startswith('-'):
                kwargs_arg = {'arg': key}
            else:
                key_with_dashes = key.replace('_', '-')
                kwargs_dest['dest'] = key
                if len(key) == 1:
                    kwargs_arg = {'arg': '-' + key_with_dashes}
                else:
                    kwargs_arg = {'arg': '--' + key_with_dashes}
            flag_valid_argument = False
            for kwargs_argparser in [kwargs_dest, kwargs_arg]:
                if flag_valid_argument:
                    continue
                ret = plant.get_args_from_argparser(parser,
                                                    store_true_action=False,
                                                    store_false_action=False,
                                                    store_action=True,
                                                    help_action=False,
                                                    **kwargs_argparser)
                if ret:
                    kwargs_str += f' {ret[0]} {value_str}'
                    flag_valid_argument = True
                    continue

                ret_store_true = plant.get_args_from_argparser(
                    parser,
                    store_true_action=True,
                    store_false_action=False,
                    store_action=False,
                    help_action=False,
                    **kwargs_argparser)
                if ret_store_true and bool(value):
                    kwargs_str += f' {ret_store_true[0]}'
                elif ret_store_true:
                    dest_store_true = plant.get_args_from_argparser(
                        parser,
                        store_true_action=True,
                        store_false_action=False,
                        store_action=False,
                        help_action=False,
                        get_dest=True,
                        **kwargs_argparser)
                    arg_store_false = plant.get_args_from_argparser(
                        parser,
                        store_true_action=False,
                        store_false_action=True,
                        store_action=False,
                        help_action=False,
                        dest=dest_store_true[0])
                    if arg_store_false:
                        kwargs_str += f' {arg_store_false[0]}'

                if ret_store_true:
                    flag_valid_argument = True
                    continue

                ret_store_false = plant.get_args_from_argparser(
                    parser,
                    store_true_action=False,
                    store_false_action=True,
                    store_action=False,
                    help_action=False,
                    **kwargs_argparser)

                if ret_store_false and bool(value):
                    kwargs_str += f' {ret_store_false[0]}'

                elif ret_store_false:
                    dest_store_false = plant.get_args_from_argparser(
                        parser,
                        store_true_action=False,
                        store_false_action=True,
                        store_action=False,
                        help_action=False,
                        get_dest=True,
                        **kwargs_argparser)
                    arg_store_true = plant.get_args_from_argparser(
                        parser,
                        store_true_action=True,
                        store_false_action=False,
                        store_action=False,
                        help_action=False,
                        dest=dest_store_false[0])
                    if arg_store_true:
                        kwargs_str += f' {arg_store_true[0]}'

                if ret_store_false:
                    flag_valid_argument = True

                if flag_valid_argument:
                    continue
            if not flag_valid_argument:
                print(f'ERROR invalid argument: "{key}"')
                return

        self._command = (f'{self._module_name}.py {args_str} {kwargs_str}'
                         f' {output_str}')

    def _update_args_str(self, args, args_str=''):
        for arg in args:
            if isinstance(arg, str):
                args_str += f' {arg}'
                continue
            if (isinstance(arg, Sequence) and
                    all([isinstance(x, str) for x in arg])):
                args_str += self._update_args_str(arg)
                continue

            if not isinstance(arg, plant.PlantImage):
                arg = plant.PlantImage(arg)

            arg_id = str(id(arg))
            plant.plant_config.variables[arg_id] = arg

            args_str += f' MEM:{arg_id}'
        return args_str
