#!/usr/bin/env python3

import os
from zipfile import ZipFile
import time
import plant
import plant_isce3
import subprocess
import numpy as np
from osgeo import gdal, ogr, osr
import copy
from datetime import datetime, timedelta
import shutil

import h5py
import boto3
import glob
import pickle
import isce3

from plant_isce3.readers.product import open_product

res_deg_dict = {'A': 1.0 / 3600,
                'B': 1.0 / 3600}

cog_str = 'COG : resampling_algorithm=average : overviews_list=2,4,8,16,32,64,128'


def get_parser():

    descr = ''
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=1,
                            geo=1,

                            aws_credentials=1,
                            default_output_options=1,
                            output_skip_if_existent=1,
                            default_flags=1,
                            output_format=1,
                            multilook=1,
                            output_dir=1,
                            bbox=1,
                            separate=1,
                            output_file=1)

    plant_isce3.add_arguments(parser,
                              product_type=1,
                              nlooks_by_frequency=1,
                              track_number=1,
                              orbit_direction=1,
                              frame_number=1,
                              cycle_number=1,
                              track_frame_db=1,
                              frequency=1)

    parser.add_argument('--bucket-name',
                        type=str,
                        dest='bucket_name',
                        help=('S3 bucket name where the input NISAR products'
                              ' are located'))

    parser.add_argument('--anc-bucket-name',
                        type=str,
                        dest='anc_bucket_name',
                        help='Ancillary files bucket name')

    parser.add_argument('--pol-mode-freq-a',
                        type=str,
                        dest='pol_mode_freq_a',
                        help=('Polarization mode for frequency A'
                              ' (e.g. "SH, DH, QP, etc.")'))
    parser.add_argument('--pol-mode-freq-b',
                        type=str,
                        dest='pol_mode_freq_b',
                        help=('Polarization mode for frequency B'
                              ' (e.g. "SH, DH, QP, etc.")'))

    parser.add_argument('--pol-list',
                        dest='pol_list',
                        nargs='+',
                        type=str,
                        help='Polarization list')

    parser.add_argument('--filename-must-include',
                        type=str,
                        nargs='*',
                        dest='filename_must_include',
                        help='List of strings that the input products should'
                        ' include (at least one)')

    parser.add_argument('--filename-must-include-all',
                        type=str,
                        nargs='*',
                        dest='filename_must_include_all',
                        help='List of strings that the input products should'
                        ' include (all)')

    parser.add_argument('--filename-must-not-include',
                        type=str,
                        nargs='*',
                        dest='filename_must_not_include',
                        help=('List of strings that the input products should'
                              ' not include'))

    parser.add_argument('--must-be-quad-pol',
                        action='store_true',
                        dest='must_be_quad_pol',
                        help='Require products to be quad-polarized')

    parser.add_argument('--must-be-mixed-mode',
                        action='store_true',
                        dest='must_be_mixed_mode',
                        help='Require products to be mixed mode')

    parser.add_argument('--must-not-be-mixed-mode',
                        '--exclude-mixed-mode',
                        action='store_true',
                        dest='must_not_be_mixed_mode',
                        help='Require products not to be mixed mode')

    parser.add_argument('--step-1-load-pickle-files',
                        action='store_true',
                        dest='step_1_load_pickle_files',
                        help='Load pickle files (if available)')

    parser.add_argument('--skip-step-1-and-load-cogs-from',

                        type=str,
                        dest='skip_step_1_and_load_cogs_from',
                        help='Skip step 1 and load pre-processed '
                        'Cloud-Optimized GeoTIFFs from this directory')

    parser.add_argument('--step-1-save-pickle-files',
                        action='store_true',
                        dest='step_1_save_pickle_files',
                        help='Save pickle files')

    parser.add_argument('--step-1-download-hdf5',
                        action='store_true',
                        dest='step_1_download_hdf5',
                        help='Download products (HDF5 files)')

    parser.add_argument('--step-1-download-png',
                        '--step-1-download-browse',
                        action='store_true',
                        dest='step_1_download_browse_png',
                        help='Download browse images (PNG files)')

    parser.add_argument('--step-1-download-kml',
                        action='store_true',
                        dest='step_1_download_kml',
                        help='Download browse KML files')

    parser.add_argument('--output-mosaic-kmz-from-all-browse-files',

                        type=str,
                        dest='step_1_mosaic_kmz',
                        help=('Name of the mosaic KMZ file from all browse'
                              ' files'))

    parser.add_argument('--output-files-prefix',
                        type=str,
                        default='',
                        dest='output_files_prefix',
                        help='Prefix for output files')

    parser.add_argument('--output-files-suffix',
                        type=str,
                        default='',
                        dest='output_files_suffix',
                        help='suffix for output files')

    parser.add_argument('--step-1-delete-files-after-mosaic',
                        action='store_true',
                        dest='step_1_delete_files_after_mosaic',
                        help=('Delete browse files after generating mosaic'
                              ' KMZ'))

    parser.add_argument('--step-2-gcov-runconfig',
                        '--step-2-generate-gcov-runconfig',
                        action='store_true',
                        dest='step_2_generate_gcov_runconfig',
                        help='Generate run configuration files')

    parser.add_argument('--step-2-tec',
                        '--step-2-download-tec',
                        action='store_true',
                        dest='step_2_download_tec_file',
                        help='Download TEC file')

    parser.add_argument('--step-2-off-diagonal-analysis',
                        action='store_true',
                        dest='step_2_off_diagonal_analysis',
                        help='Perform off-diagonal analysis')

    parser.add_argument('--worldcover',
                        '--worldcover-file',
                        dest='worldcover',
                        type=str,
                        help='WorldCover land cover map for EAP analysis')

    parser.add_argument('--step-2-eap-analysis',
                        action='store_true',
                        dest='step_2_eap_analysis',
                        help='Perform EAP analysis')

    parser.add_argument('--full-covariance',
                        '--fullcovariance',
                        dest='full_covariance',
                        action='store_true',
                        help='Include off-diagonal terms in the covariance'
                        ' matrix.')

    parser.add_argument('--plot-date',
                        type=str,
                        dest='plot_date',
                        help='Date string to include in plot titles')

    parser.add_argument('--plot-dataset-name',
                        type=str,
                        dest='plot_dataset_name',
                        help='Dataset name to include in plot titles')

    parser.add_argument('--n-parallel-processes',
                        type=int,
                        default=8,
                        dest='n_parallel_processes',
                        help=('Maximum number of parallel processes (if'
                              ' applicable)'))

    parser.add_argument('--step-2-generate-cog',
                        action='store_true',
                        dest='step_2_generate_cog',
                        help='Generate Cloud-Optimized GeoTIFFs (COGs)')

    parser.add_argument('--step-2-generate-cog-parallel',
                        action='store_true',
                        dest='step_2_generate_cog_parallel',
                        help=('Generate Cloud-Optimized GeoTIFFs (COGs) in'
                              ' parallel'))

    parser.add_argument('--plant-isce3-util-path',
                        type=str,
                        default='plant_isce3_util.py',
                        dest='plant_isce3_util_path',
                        help=('Path to plant_isce3_util.py script'
                              ' (for parallel processing)'))

    parser.add_argument('--step-2-generate-cog-rgb',
                        action='store_true',
                        dest='step_2_generate_cog_rgb',
                        help='Generate Cloud-Optimized GeoTIFFs (COGs) RGB-'
                        'color composite')

    parser.add_argument('--step-2-generate-kmz',
                        action='store_true',
                        dest='step_2_generate_kmz',
                        help='Generate KMZ files')

    parser.add_argument('--step-2-generate-png',
                        action='store_true',
                        dest='step_2_generate_png',
                        help='Generate PNG files')

    parser.add_argument('--step-2-generate-all-layers-kmz',
                        action='store_true',
                        dest='step_2_generate_all_layers_kmz',
                        help='Generate KMZ files from all available layers')

    parser.add_argument('--step-2-generate-all-layers-png',
                        action='store_true',
                        dest='step_2_generate_all_layers_png',
                        help='Generate PNG files from all available layers')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--masked-data',
                       '--masked-images',
                       dest='masked_data_file',
                       action='store_true',
                       help=("Extract product's imagery and apply valid data"
                             " mask"))

    group.add_argument('--rtc-gamma-to-sigma',
                       '--rtc-gamma-to-sigma-layer',
                       dest='rtc_gamma_to_sigma_file',
                       action='store_true',
                       help=("Extract the RTC gamma to sigma layer from the"
                             " product"))

    group.add_argument('--number-of-looks',
                       '--number-of-looks-layer',
                       dest='number_of_looks_file',
                       action='store_true',
                       help=("Extract the RTC gamma to sigma layer from the"
                             " product"))

    group.add_argument('--eap',
                       '--elevation-antenna-patter',
                       dest='elevation_antenna_pattern_file',
                       action='store_true',
                       help=("Extract elevation antenna pattern (EAP) layer."))

    group.add_argument('--neb',
                       '--noise-equivalent-backscatter',
                       dest='noise_equivalent_backscatter_file',
                       action='store_true',
                       help=("Extract noise equivalent backscatter layer."))

    group.add_argument('--doppler-centroid',
                       dest='doppler_centroid_file',
                       action='store_true',
                       help=("Extract Doppler centroid layer."))

    parser.add_argument('--step-3-generate-vrt',
                        action='store_true',
                        dest='step_3_generate_vrt',
                        help='Generate VRT projection files')

    parser.add_argument('--step-4-generate-tiles',
                        action='store_true',
                        dest='step_4_generate_tiles',
                        help='Generate mosaic tiles')

    parser.add_argument('--step-4-generate-tiles-parallel',
                        action='store_true',
                        dest='step_4_generate_tiles_parallel',
                        help='Generate mosaic tiles in parallel')

    parser.add_argument('--plant-mosaic-path',
                        type=str,
                        default='plant_mosaic.py',
                        dest='plant_mosaic_path',
                        help=('Path to plant_mosaic.py script'
                              ' (for parallel processing)'))

    parser.add_argument('--step-4-generate-tiles-kmz',
                        action='store_true',
                        dest='step_4_generate_tiles_kmz',
                        help='Generate mosaic KMZ files')

    parser.add_argument('--step-4-generate-tiles-rgb-kmz',
                        action='store_true',
                        dest='step_4_generate_tiles_rgb_kmz',
                        help='Generate mosaic RGB-color composition KMZ files')

    parser.add_argument('--step-4-generate-tiles-ab-kmz',
                        action='store_true',
                        dest='step_4_generate_tiles_ab_kmz',
                        help='Generate dual-frequency (A and B) '
                        'mosaic KMZ files')

    parser.add_argument('--step-5-generate-mosaic-vrt',
                        action='store_true',
                        dest='step_5_generate_mosaic_vrt',
                        help='Generate mosaic VRT files')

    parser.add_argument('--step-6-generate-mosaic-kmz',
                        action='store_true',
                        dest='step_6_generate_mosaic_kmz',
                        help='Generate mosaic KMZ files')

    parser.add_argument('--step-6-generate-mosaic-pol-kmz',
                        action='store_true',
                        dest='step_6_generate_mosaic_pol_kmz',
                        help=('Generate mosaic KMZ files of a single-'
                              'polarization'))

    parser.add_argument('--step-7-generate-mosaic-ab-kmz',
                        action='store_true',
                        dest='step_7_generate_mosaic_ab_kmz',
                        help=('Generate mosaic dual-frequency (A and B) KMZ'
                              ' files'))

    parser.add_argument('--step-8-generate-tile-map-kmz',
                        action='store_true',
                        dest='step_8_generate_time_map_kmz',
                        help='Generate tile map KMZ file')

    parser.add_argument('--max-number-products',
                        type=int,
                        dest='max_number_products',
                        help='Maximum number of products to process')

    parser.add_argument('--cache-directory',
                        type=str,
                        default='0_cached_files',
                        dest='cache_directory',
                        help='Cache directory')

    parser.add_argument('--step-1-directory',
                        '--step-1-downloaded-data-directory',
                        '--downloaded-data-directory',
                        type=str,
                        default='1_downloaded_data',
                        dest='step_1_directory',
                        help='Downloaded data directory')

    parser.add_argument('--step-2-directory',
                        '--step-2-processed-files-native-grid-directory',
                        type=str,
                        default='2_processed_files_native_grid',
                        dest='step_2_directory',
                        help='Directory containing processed files under'
                        " the product's native grid")

    parser.add_argument('--step-3-directory',
                        '--step-3-mosaic-files-native-grid-directory',
                        type=str,
                        default='3_mosaic_files_native_grid',
                        dest='step_3_directory',
                        help='Directory containing mosaic files under'
                        " the product's native grid")

    parser.add_argument('--step-4-directory',
                        '--step-4-processed-files-geographic-directory',
                        '--processed-files-geographic-directory',
                        type=str,
                        default='4_processed_files_geographic',
                        dest='step_4_directory',
                        help='Directory containing processed files in'
                        ' geographic coordinates')

    parser.add_argument('--step-5-directory',
                        '--step-5-mosaic-files-geographic-directory',
                        type=str,
                        default='5_mosaic_files_geographic',
                        dest='step_5_directory',
                        help='Directory containing mosaic files in'
                        ' geographic coordinates')

    parser.add_argument('--profile-max-in-db',
                        type=float,
                        dest='profile_max_in_db',
                        help='Maximum value for EAP profiles in dB')

    parser.add_argument('--profile-min-in-db',
                        type=float,
                        dest='profile_min_in_db',
                        help='Minimum value for EAP profiles in dB')

    parser.add_argument('--create-plots-with-predefined-thresholds',
                        action='store_true',
                        dest='flag_create_plots_with_predefined_thresholds',
                        help=('Create plots with predefined thresholds for'
                              ' better visual comparison between profiles'))

    return parser


def parse_tec_filename(tec_filename):
    tec_basename = os.path.basename(tec_filename)
    creation_datetime = datetime.strptime(tec_basename[14:14 + 15],
                                          "%Y%m%dT%H%M%S")
    start_datetime = datetime.strptime(tec_basename[30:30 + 15],
                                       "%Y%m%dT%H%M%S")
    end_datetime = datetime.strptime(tec_basename[46:46 + 15],
                                     "%Y%m%dT%H%M%S")
    ret_dict = {}
    ret_dict['creation_datetime'] = creation_datetime
    ret_dict['start_datetime'] = start_datetime
    ret_dict['end_datetime'] = end_datetime
    return ret_dict


class PlantIsce3BatchProcessing(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        self.replace_null = False
        super().__init__(parser, argv)

    def run(self):

        if (self.step_2_generate_cog_parallel and
                not self.plant_isce3_util_path):
            self.print('ERROR: --plant-isce3-util-path is required for'
                       ' --step-2-generate-cog-parallel')
            return

        if (self.step_4_generate_tiles_parallel and
                not self.plant_mosaic_path):
            self.print('ERROR: --plant-mosaic-path is required for'
                       ' --step-4-generate-tiles-parallel')
            return

        if (not self.dem_file and
            (self.step_2_generate_gcov_runconfig or

             self.step_2_eap_analysis)):
            self.print('ERROR: --dem-file is required for the selected'
                       ' processing steps')
            return

        aws_credentials_file = f"{os.path.expanduser('~')}/.aws/credentials"
        gdal.SetConfigOption('AWS_CONFIG_FILE', aws_credentials_file)

        bucket_name = None
        if self.bucket_name:
            flag_s3_bucket = True
            bucket_name = self.bucket_name.replace('s3://', '')
        else:
            input_file_splitted = self.input_file.split('/')
            flag_s3_bucket = input_file_splitted[0] == 's3:'
            if (flag_s3_bucket and len(input_file_splitted) > 1 and
                    input_file_splitted[1] != ''):
                self.print(f'ERROR invalid s3 path: {input_file_splitted}')
                return

        self.print(
            f'process files from s3 bucket (True/False): {flag_s3_bucket}')

        if self.product_type:
            self.print(f'product type: {self.product_type}')
            if self.product_type == 'RRSD':
                product_level = 'L0B'
            elif self.product_type == 'RSLC':
                product_level = 'L1'
            elif self.product_type in ['GCOV', 'GSLC', 'STATIC']:
                product_level = 'L2'
            else:
                print('ERROR product type not supported:'
                      f' {self.product_type}')

        if flag_s3_bucket:
            if not bucket_name:
                bucket_name = input_file_splitted[2]
            self.print(f's3 bucket: {bucket_name}')
            if self.input_file and len(input_file_splitted) > 3:
                s3_prefix = '/'.join(input_file_splitted[3:])
            else:
                s3_prefix = ''
            self.print(f's3 prefix: {s3_prefix}')

            with plant.PlantIndent():
                self.print(f'product type: {self.product_type}')
                self.print(f'cycle number: {self.cycle_number}')
                self.print(f'track number: {self.track_number}')
                self.print(f'frame number: {self.frame_number}')
                self.print(f'track frame database: {self.track_frame_db}')

            if (not s3_prefix and
                    not bucket_name.startswith('sds-n-cumulus-prod-nisar-products') and
                    self.product_type and
                    self.cycle_number and
                    self.track_number and
                    self.frame_number and
                    self.track_frame_db):

                date_str = \
                    plant_isce3.get_nisar_track_frame_date_from_tfdb(
                        self.track_number,
                        self.frame_number,
                        self.track_frame_db,
                        self.cycle_number)

                s3_prefix = (f'products/{product_level}_L'
                             f'_{self.product_type}/{date_str}')
                self.print(f's3 prefix (updated): {s3_prefix}')

            elif (not s3_prefix and
                  not bucket_name.startswith('sds-n-cumulus-prod-nisar-products') and
                    self.product_type):

                s3_prefix = f'products/{product_level}_L_{self.product_type}'
                self.print(f's3 prefix (updated): {s3_prefix}')

            elif (not s3_prefix and
                    not bucket_name.startswith('sds-n-cumulus-prod-nisar-products')):
                print('ERROR could not determine the s3 path')
                return

        else:
            bucket_name = None
            s3_prefix = None

        kwargs_color_orig_dict = {
            'cmap': self.cmap,

            'cmap_crop_min': self.cmap_crop_min,
            'background_color': self.background_color,
            'percentile': self.percentile,
            'flag_add_kmz_cbar_offset': self.flag_add_kmz_cbar_offset,
            'kmz_cbar_offset_color': self.kmz_cbar_offset_color,
            'kmz_cbar_offset_length': self.kmz_cbar_offset_length,
            'kmz_cbar_offset_width': self.kmz_cbar_offset_width,
            'kmz_cbar_offset_alpha': self.kmz_cbar_offset_alpha
        }
        if self.cmap_min is not None and isinstance(self.cmap_min, list):
            kwargs_color_orig_dict['cmap_min'] = ','.join([
                str(c) for c in self.cmap_min])
        else:
            kwargs_color_orig_dict['cmap_min'] = self.cmap_min

        if self.cmap_max is not None and isinstance(self.cmap_max, list):
            kwargs_color_orig_dict['cmap_max'] = ','.join([
                str(c) for c in self.cmap_max])
        else:
            kwargs_color_orig_dict['cmap_max'] = self.cmap_max

        kwargs_color = {}
        for k, v in kwargs_color_orig_dict.items():
            if v is None:
                continue
            kwargs_color[k] = v

        tiles_map_by_epsg = {}
        bbox_by_epsg = {}
        frequency_epsg_dict_pickle_file = \
            (f'pickle_files/{self.output_files_prefix}'
             f'frequency_epsg_dict{self.output_files_suffix}.pkl')
        tiles_map_by_epsg_pickle_file = \
            (f'pickle_files/{self.output_files_prefix}'
             f'tiles_map_by_epsg{self.output_files_suffix}.pkl')
        bbox_by_epsg_pickle_file = \
            (f'pickle_files/{self.output_files_prefix}'
             f'bbox_by_epsg{self.output_files_suffix}.pkl')

        if self.skip_step_1_and_load_cogs_from:

            print('Step 1: loading COG files from directory:'
                  f' {self.skip_step_1_and_load_cogs_from}')

            search_pattern = self.skip_step_1_and_load_cogs_from

            file_list = glob.glob(search_pattern, recursive=True)

            if len(file_list) == 0:

                search_pattern = os.path.join(
                    self.skip_step_1_and_load_cogs_from, '**', '*.tif')

            if len(file_list) == 0:
                print('ERROR no COG files found with pattern:',
                      self.skip_step_1_and_load_cogs_from)
                return

            frequency_epsg_dict = {}

            n_files = len(file_list)

            cycle_list = []
            track_frame_list = []
            products_list = []
            products_pol_dict = {}
            pol_modes_dict = {}

            for i, tif_file in enumerate(file_list):
                plant.print_progress(i, n_files)

                print(f'*** evaluating file {tif_file}')
                try:
                    dict_filename = self.parse_nisar_product_filename(tif_file)

                except BaseException:
                    dict_filename = None

                print('*** filename parsing result:', dict_filename)

                if not self.meet_filename_requirements('', tif_file,
                                                       dict_filename):
                    continue

                print('*** filename meets requirements, processing file')

                image_obj = plant.read_image(tif_file)
                metadata = image_obj.metadata
                if metadata is None:
                    print('====================================')
                    print('====================================')
                    print('====================================')
                    print(f'WARNING No metadata found for file: {tif_file}.'
                          ' Skipping.')
                    print(f'WARNING No metadata found for file: {tif_file}.'
                          ' Skipping.')
                    print(f'WARNING No metadata found for file: {tif_file}.'
                          ' Skipping.')
                    print('====================================')
                    print('====================================')
                    print('====================================')
                    continue

                frequency = None
                pol = None
                bounding_polygon_wkt = None
                if metadata is None:
                    print('tif_file:', tif_file)
                    raise ValueError('TIF file has no valid metadata:'
                                     f' {tif_file}')
                for key, value in metadata.items():

                    if key == 'FREQUENCY':
                        frequency = value
                        print('*** frequency:', frequency)
                        continue
                    if key == 'POLARIZATION':
                        pol = value
                        print('*** polarization:', pol)
                    if key == 'BOUNDING_POLYGON':
                        bounding_polygon_wkt = value
                        print('*** bounding polygon:', bounding_polygon_wkt)

                if (frequency is None and dict_filename is not None and
                        'frequency' in dict_filename.keys()):
                    frequency = dict_filename['frequency']
                    print('*** frequency from filename:', frequency)

                if (pol is None and dict_filename is not None and
                        'polarization' in dict_filename.keys()):
                    pol = dict_filename['polarization']
                    print('*** polarization from filename:', pol)

                if frequency is None or pol is None:
                    print(f'***    Unrecognized file: {tif_file}. Skipping.')
                    continue

                if self.pol_list is not None and pol not in self.pol_list:
                    print(f'***        skipping polarization {pol} based on'
                          ' user input')
                    continue

                epsg = image_obj.geogrid.epsg
                print('*** epsg:', epsg)

                if bounding_polygon_wkt is None:
                    length = image_obj.length
                    width = image_obj.width
                    geotransform = image_obj.geotransform
                    x0 = geotransform[0]
                    y0 = geotransform[3]
                    xf = x0 + geotransform[1] * width
                    yf = y0 + geotransform[5] * length
                    bounding_polygon = ogr.Geometry(ogr.wkbPolygon)
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    ring.AddPoint(x0, y0)
                    ring.AddPoint(xf, y0)
                    ring.AddPoint(xf, yf)
                    ring.AddPoint(x0, yf)
                    ring.AddPoint(x0, y0)
                    bounding_polygon.AddGeometry(ring)

                    src_srs = osr.SpatialReference()
                    src_srs.ImportFromEPSG(epsg)
                    src_srs.SetAxisMappingStrategy(
                        osr.OAMS_TRADITIONAL_GIS_ORDER)

                    dst_srs = osr.SpatialReference()
                    dst_srs.ImportFromEPSG(4326)
                    dst_srs.SetAxisMappingStrategy(
                        osr.OAMS_TRADITIONAL_GIS_ORDER)

                    transform = osr.CoordinateTransformation(src_srs, dst_srs)

                    bounding_polygon.AssignSpatialReference(src_srs)
                    bounding_polygon.Transform(transform)

                    bounding_polygon_wkt = bounding_polygon.ExportToWkt()

                    print('*** bounding polygon in WKT format:'
                          f' {bounding_polygon_wkt}')

                min_lon, max_lon, min_lat, max_lat, center_lon, center_lat = \
                    self.get_polygon_parameters(bounding_polygon)

                print('*** center_lon:', center_lon)

                if self.bbox:

                    outer_polygon_ogr = self.get_bbox_polygon()

                    self.print('***        product extents:')
                    with plant.PlantIndent():
                        self.print(f'***            min_lat: {min_lat}')
                        self.print(f'***            min_lon: {min_lon}')
                        self.print(f'***            max_lat: {max_lat}')
                        self.print(f'***            max_lon: {max_lon}')

                    if not outer_polygon_ogr.Intersects(bounding_polygon):
                        print('***    Product does not intersect with the'
                              ' selection bbox. Skipping.')
                        continue

                    bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon = \
                        self.bbox
                    self.print('***        selection bbox:')
                    with plant.PlantIndent():
                        self.print('***             bbox_min_lat:'
                                   f' {bbox_min_lat}')
                        self.print('***             bbox_min_lon:'
                                   f' {bbox_min_lon}')
                        self.print('***             bbox_max_lat:'
                                   f' {bbox_max_lat}')
                        self.print('***             bbox_max_lon:'
                                   f' {bbox_max_lon}')

                    if not outer_polygon_ogr.Intersects(bounding_polygon):
                        print('***    Product does not intersect with the'
                              ' selection bbox. Skipping.')
                        continue

                self.update_tiles_map_dict(tiles_map_by_epsg, bbox_by_epsg,
                                           bounding_polygon, epsg)

                if frequency not in frequency_epsg_dict.keys():
                    frequency_epsg_dict[frequency] = {
                        pol: {epsg: [tif_file]}
                    }
                if pol not in frequency_epsg_dict[frequency].keys():
                    frequency_epsg_dict[frequency][pol] = {
                        epsg: [tif_file]
                    }
                elif epsg not in frequency_epsg_dict[frequency][pol].keys():
                    frequency_epsg_dict[frequency][pol][epsg] = \
                        [tif_file]
                else:
                    frequency_epsg_dict[frequency][pol][epsg].append(
                        tif_file)

                if dict_filename is not None:
                    cycle_number = None
                    track_frame = None
                    orbit_direction = None
                    if 'cycle_number' in dict_filename.keys():
                        cycle_number = dict_filename['cycle_number']
                        if cycle_number not in cycle_list:
                            cycle_list.append(cycle_number)
                    if ('track_number' in dict_filename.keys() and
                            'frame_number' in dict_filename.keys()):
                        track_frame = \
                            (f"{dict_filename['track_number']}_"
                             f"{dict_filename['orbit_direction_char']}_"
                             f"{dict_filename['frame_number']}")
                        if track_frame not in track_frame_list:
                            track_frame_list.append(track_frame)

                    if cycle_number is not None or track_frame is not None:
                        cycle_track_frame_str = f"{cycle_number}_{track_frame}"
                        if cycle_track_frame_str not in products_list:
                            products_list.append(cycle_track_frame_str)
                        if pol not in products_pol_dict:
                            products_pol_dict[pol] = []
                        if cycle_track_frame_str not in products_pol_dict[pol]:
                            products_pol_dict[pol].append(
                                cycle_track_frame_str)
                        pol_mode_a = dict_filename.get('pol_freq_a', None)
                        if pol_mode_a is not None:
                            if pol_mode_a not in pol_modes_dict:
                                pol_modes_dict[pol_mode_a] = []
                            if cycle_track_frame_str not in \
                                    pol_modes_dict[pol_mode_a]:
                                pol_modes_dict[pol_mode_a].append(
                                    cycle_track_frame_str)
                        pol_mode_b = dict_filename.get('pol_freq_b', None)
                        if pol_mode_b is not None:
                            if pol_mode_b not in pol_modes_dict:
                                pol_modes_dict[pol_mode_b] = []
                            if cycle_track_frame_str not in \
                                    pol_modes_dict[pol_mode_b]:
                                pol_modes_dict[pol_mode_b].append(
                                    cycle_track_frame_str)

            print('list of products (cycle_track_frame):', products_list)
            print('list of track-frames:', track_frame_list)
            print('list of cycles:', cycle_list)
            print('list of pol modes:', pol_modes_dict)
            print('')
            print('number of products (cycle_track_frame):',
                  len(products_list))
            for pol_mode, product_list in pol_modes_dict.items():
                print(f'number of products for polarization mode {pol_mode}:'
                      f' {len(product_list)}')
            print('')
            for pol, product_list in products_pol_dict.items():
                print(f'number of products for polarization {pol}:'
                      f' {len(product_list)}')
            print('')
            print('number of track-frames:', len(track_frame_list))
            print('number of cycles:', len(cycle_list))

        elif self.step_1_load_pickle_files:
            with plant.PlantIndent():
                print('Step 1: loading pickle files')
                with open(frequency_epsg_dict_pickle_file, 'rb') as \
                        pickle_file:
                    frequency_epsg_dict = pickle.load(pickle_file)

                with open(tiles_map_by_epsg_pickle_file, 'rb') as pickle_file:
                    tiles_map_by_epsg = pickle.load(pickle_file)

                with open(bbox_by_epsg_pickle_file, 'rb') as pickle_file:
                    bbox_by_epsg = pickle.load(pickle_file)

        else:
            frequency_epsg_dict, mosaic_kmz_file_list = \
                self.step_1_2_processing_native_coordinates(
                    flag_s3_bucket, bucket_name, s3_prefix, kwargs_color,
                    tiles_map_by_epsg, bbox_by_epsg)

        if self.step_1_save_pickle_files:
            os.makedirs('pickle_files', exist_ok=True)
            with open(frequency_epsg_dict_pickle_file, 'wb') as pickle_file:
                pickle.dump(frequency_epsg_dict, pickle_file)
            print('## file saved:', frequency_epsg_dict_pickle_file)

            with open(tiles_map_by_epsg_pickle_file, 'wb') as pickle_file:
                pickle.dump(tiles_map_by_epsg, pickle_file)
            print('## file saved:', tiles_map_by_epsg_pickle_file)

            with open(bbox_by_epsg_pickle_file, 'wb') as pickle_file:
                pickle.dump(bbox_by_epsg, pickle_file)
            print('## file saved:', bbox_by_epsg_pickle_file)

        if self.step_1_mosaic_kmz and len(mosaic_kmz_file_list) > 0:
            print('    Step 1: Mosaic KMZ')
            mosaic_doc_kml_file = f'{self.step_1_directory}/doc.kml'

            if plant.isfile(mosaic_doc_kml_file):
                update = self.overwrite_file_check(mosaic_doc_kml_file)
            else:
                update = True

            if update:
                with open(mosaic_doc_kml_file, 'w') as f:
                    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
                    f.write('  <Document>\n')
                    for filename_kml in mosaic_kmz_file_list:
                        if not filename_kml.endswith('.kml'):
                            continue
                        basename_kml = os.path.basename(filename_kml)
                        kml_zip_path = os.path.join('files', basename_kml)
                        basename = basename_kml.replace('.kml', '')

                        f.write('    <NetworkLink>\n')
                        f.write(f'      <name>{basename}</name>\n')
                        f.write('      <Link>\n')
                        f.write(f'        <href>{kml_zip_path}</href>\n')
                        f.write('      </Link>\n')
                        f.write('    </NetworkLink>\n')
                    f.write('  </Document>\n')
                    f.write('</kml>\n')

                if plant.isfile(self.step_1_mosaic_kmz):
                    update = self.overwrite_file_check(self.step_1_mosaic_kmz)
                else:
                    update = True

                if update:
                    with ZipFile(self.step_1_mosaic_kmz, 'w') as myzip:
                        for filename in mosaic_kmz_file_list:
                            basename = os.path.basename(filename)
                            myzip.write(filename,
                                        os.path.join('files', basename))
                        myzip.write(mosaic_doc_kml_file, 'doc.kml')
                    print(f'## file saved: {self.step_1_mosaic_kmz}')

        if (self.step_1_delete_files_after_mosaic and
                len(mosaic_kmz_file_list) > 0):
            print('    Step 1: Deleting browse files')
            for filename in mosaic_kmz_file_list:
                os.remove(filename)
                print(f'## file deleted: {filename}')

        print('## Processing steps 3-5')

        orbit_direction_str = ''

        for frequency, pol_dict in frequency_epsg_dict.items():

            suffix_list = []

            for pol, epsg_dict in pol_dict.items():
                if self.pol_list is not None and pol not in self.pol_list:
                    print(f'***        skipping polarization {pol} based on'
                          ' user input')
                    continue

                suffix = f'_{frequency}_{pol}{orbit_direction_str}'
                suffix_rgb = f'_{frequency}{orbit_direction_str}'

                suffix_list.append(suffix)

                list_of_epsg_vrts = []

                for epsg, file_list in epsg_dict.items():

                    print(f'## Processing EPSG: {epsg} ({len(file_list)})')

                    vrt_file = (f'{self.step_3_directory}/'
                                f'{self.output_files_prefix}EPSG{epsg}{suffix}'
                                f'{orbit_direction_str}'
                                f'{self.output_files_suffix}.vrt')

                    if (self.step_3_generate_vrt and
                            not os.path.isfile(vrt_file)):
                        os.makedirs(self.step_3_directory, exist_ok=True)
                        print('    Step 3: Building VRT from files:',
                              file_list, ', output file:', vrt_file)
                        if os.path.isfile(vrt_file):
                            os.remove(vrt_file)

                        gdal.BuildVRT(vrt_file, file_list, srcNodata='nan',
                                      resampleAlg='average',
                                      VRTNodata='nan')
                        print('##        file saved:', vrt_file)
                        add_overviews_vrt(vrt_file)
                        print(f'        file updated: {vrt_file}'
                              ' (added overviews)')
                    elif not os.path.isfile(vrt_file):
                        continue

                    plant_image_obj = plant.read_image(vrt_file)

                    if plant_image_obj is None:
                        print('WARNING there was an error opening VRT file:'
                              f' "{vrt_file}". Skipping.')
                        continue

                    list_of_epsg_vrts.append(vrt_file)

                flag_last_pol = pol == list(pol_dict.keys())[-1]

                vrt_file = self.run_processing_geographic(
                    tiles_map_by_epsg, bbox_by_epsg, orbit_direction_str,
                    frequency, flag_last_pol, suffix_list, suffix_rgb,
                    suffix, list_of_epsg_vrts)

            if self.step_7_generate_mosaic_ab_kmz:

                mosaic_a_vrt_file = (
                    f'{self.step_5_directory}/'
                    f'{self.output_files_prefix}mosaic_A_{pol}'
                    f'{orbit_direction_str}'
                    f'{self.output_files_suffix}.vrt')
                mosaic_b_vrt_file = (
                    f'{self.step_5_directory}/'
                    f'{self.output_files_prefix}mosaic_B_{pol}'
                    f'{orbit_direction_str}'
                    f'{self.output_files_suffix}.vrt')

                print('    Step 6: KMZ')
                kmz_file = (f'{self.step_5_directory}/'
                            f'{self.output_files_prefix}mosaic_AB_{pol}'
                            f'{orbit_direction_str}'
                            f'{self.output_files_suffix}.kmz')

                print('pol_vrt_list:', vrt_file)
                self.util(mosaic_a_vrt_file, mosaic_b_vrt_file,
                          output_file=kmz_file, force=True,

                          in_null=np.nan)

        print('done')

    def meet_filename_requirements(self, path, filename, dict_filename):
        if (self.filename_must_include is not None and
                len(self.filename_must_include) > 0):

            for s in self.filename_must_include:
                print(f'*** filename must include: {s}')
                if s in path or s in filename:

                    break
            else:

                return False

        if (self.filename_must_include_all is not None and
                len(self.filename_must_include_all) > 0):
            flag_all_found = True
            for s in self.filename_must_include_all:
                print(f'*** filename must include (all): {s}')
                if s not in path and s not in filename:
                    flag_all_found = False
                    break
            if not flag_all_found:
                return False

        if (self.filename_must_not_include is not None and
                len(self.filename_must_not_include) > 0):
            flag_found = False
            for s in self.filename_must_not_include:
                if s in path or s in filename:
                    flag_found = True
                    break
            if flag_found:
                return False

        if dict_filename is None:
            return True

        if (self.product_type is not None and
                'product_type' in dict_filename.keys()):
            if dict_filename['product_type'] != self.product_type:
                print('*** filename product type:'
                      f' {dict_filename["product_type"]},'
                      f' required product type: {self.product_type}')
                return False

        if (self.must_be_quad_pol and
                (dict_filename['pol_freq_a'] != 'QP' and
                 dict_filename['pol_freq_b'] != 'QP')):
            print('*** filename polarization modes:'
                  f' {dict_filename["pol_freq_a"]},'
                  f' {dict_filename["pol_freq_b"]},'
                  ' required: QP')
            return False

        if (self.must_be_mixed_mode and
                dict_filename['radar_processing_mode'] != 'M'):

            return False

        if (self.must_not_be_mixed_mode and
                dict_filename['radar_processing_mode'] == 'M'):

            return False

        if (self.cycle_number is not None and
                dict_filename['cycle_number'] !=
                int(self.cycle_number)):
            print('*** filename cycle number:'
                  f' {dict_filename["cycle_number"]},'
                  f' required cycle number: {self.cycle_number}')
            return False

        if (self.orbit_direction is not None and
                dict_filename['orbit_direction'] !=
                self.orbit_direction):
            print('*** filename orbit pass direction:'
                  f' {dict_filename["orbit_direction"]},'
                  f' required orbit pass direction: {self.orbit_direction}')
            return False

        if (self.track_number is not None and
                'track_number' in dict_filename.keys() and
                dict_filename['track_number'] != int(self.track_number)):
            print('*** filename frame number:'
                  f' {dict_filename["track_number"]},'
                  f' required frame number: {self.track_number}')
            return False

        if (self.frame_number is not None and
                'frame_number' in dict_filename.keys() and
                dict_filename['frame_number'] != int(self.frame_number)):
            print('*** filename frame number:'
                  f' {dict_filename["frame_number"]},'
                  f' required frame number: {self.frame_number}')
            return False

        if (self.pol_mode_freq_a is not None and
                dict_filename['pol_freq_a'] != self.pol_mode_freq_a):
            print('*** filename polarization mode for frequency A:'
                  f' {dict_filename["pol_freq_a"]},'
                  f' required polarization mode: {self.pol_mode_freq_a}')
            return False
        if (self.pol_mode_freq_b is not None and
                dict_filename['pol_freq_b'] != self.pol_mode_freq_b):
            print('*** filename polarization mode for frequency B:'
                  f' {dict_filename["pol_freq_b"]},'
                  f' required polarization mode: {self.pol_mode_freq_b}')
            return False

        return True

    def run_processing_geographic(
            self, tiles_map_by_epsg, bbox_by_epsg, orbit_direction_str,
            frequency, flag_last_pol, suffix_list, suffix_rgb,
            suffix, list_of_epsg_vrts):
        mosaic_tiles_map = tiles_map_by_epsg['mosaic']
        mosaic_min_lon, mosaic_max_lon, mosaic_min_lat, mosaic_max_lat = \
            bbox_by_epsg['mosaic']

        vrt_file = self.create_tiles(
            self.step_4_generate_tiles,
            self.step_4_generate_tiles_parallel,
            self.step_4_generate_tiles_kmz,
            self.step_4_generate_tiles_rgb_kmz,
            self.step_4_generate_tiles_ab_kmz,
            self.step_5_generate_mosaic_vrt,
            frequency, orbit_direction_str,
            mosaic_min_lat, mosaic_max_lat,
            mosaic_min_lon, mosaic_max_lon,
            mosaic_tiles_map,
            flag_last_pol, suffix_list, suffix_rgb,

            list_of_epsg_vrts,

            output_dir_prefix=self.step_4_directory,
            suffix=suffix)

        if self.step_6_generate_mosaic_kmz:
            print('    Step 6: Kmz')
            kmz_file = (f'{self.step_5_directory}/'
                        f'{self.output_files_prefix}mosaic'
                        f'{suffix}{self.output_files_suffix}.kmz')
            self.util(vrt_file, output_file=kmz_file, force=True,

                      in_null=np.nan)

        if self.step_6_generate_mosaic_pol_kmz:
            for pol_count, pol in enumerate(['_HH', '_HV']):
                mosaic_vrt_file = (
                    f'{self.step_5_directory}/'
                    f'{self.output_files_prefix}mosaic'
                    f'{suffix}{self.output_files_suffix}.vrt'
                )
                print('    Step 6: Kmz')
                kmz_file = (f'{self.step_5_directory}/'
                            f'{self.output_files_prefix}mosaic'
                            f'{suffix}{self.output_files_suffix}.kmz')
                print('vrt_file:', vrt_file)
                self.util(mosaic_vrt_file, output_file=kmz_file,
                          force=True,

                          in_null=np.nan)

        if self.step_8_generate_time_map_kmz:
            tiles_map_geotransform = [-180, 1, 0, 90, 0, -1]

            plant.save_image(
                mosaic_tiles_map.copy(),
                f'{self.step_5_directory}/{self.output_files_prefix}'
                f'tiles_map{suffix}{self.output_files_suffix}.kmz',
                geotransform=tiles_map_geotransform,
                force=True)
            for epsg, tile_map in tiles_map_by_epsg.items():
                if epsg == 'mosaic':
                    continue
                plant.save_image(
                    tile_map.copy(),
                    f'{self.step_5_directory}/{self.output_files_prefix}'
                    f'tiles_map_{epsg}_{suffix}{self.output_files_suffix}.kmz',
                    geotransform=tiles_map_geotransform,
                    force=True)

        return vrt_file

    def step_1_2_processing_native_coordinates(
            self, flag_s3_bucket, bucket_name, s3_prefix, kwargs_color,

            tiles_map_by_epsg, bbox_by_epsg):

        product_count = 1
        frequency_epsg_dict = {'A': {}, 'B': {}}

        mosaic_kmz_file_list = []

        if flag_s3_bucket:

            print(f'    Step 1: loading datasets from s3://{bucket_name}/'
                  f'{s3_prefix}')

            creds = plant.load_aws_credentials()

            resource = boto3.resource('s3', **creds)

            nisar_product_bucket = resource.Bucket(bucket_name)

            files_iterator = nisar_product_bucket.objects.filter(
                Prefix=s3_prefix)

        else:
            print('    Step 1: loading datasets from file(s):'
                  f' {self.input_file}')
            file_list = glob.glob(self.input_file, recursive=True)
            files_iterator = [os.path.split(f) for f in file_list]

        previous_date_str = None
        cycle_list = []

        for i, objects in enumerate(files_iterator):

            if isinstance(objects, tuple):
                path, f = objects
                if not path:
                    path = '.'
            else:
                path, f = os.path.split(objects.key)

            basename = os.path.splitext(f)[0]

            if not f.endswith('.h5') or 'STATS' in f:
                continue

            print('***    f:', f)
            print('***    path:', path)

            try:
                dict_filename = self.parse_nisar_product_filename(f)

            except BaseException:
                dict_filename = None

            try:
                dict_path = self.parse_nisar_s3_path(path)
            except BaseException:
                dict_path = None

            if dict_path is not None:
                date_str = (f"{dict_path['year']}-{dict_path['month']}"
                            f"-{dict_path['day']}")
                if dict_filename is not None:
                    cycle_number = dict_filename['cycle_number']
                    if cycle_number not in cycle_list:
                        cycle_list.append(cycle_number)

                if date_str != previous_date_str:
                    print(f'## Processing date: {date_str}')
                    if len(cycle_list) > 0:
                        print(f'    Available orbit cycles: {cycle_list}')
                    previous_date_str = date_str
                    cycle_list = []

            if not self.meet_filename_requirements(path, f, dict_filename):
                continue

            cache_hdf5 = None
            if flag_s3_bucket:
                downloaded_file = os.path.join(self.step_1_directory, f)
                if (not os.path.isfile(downloaded_file) and
                        not self.step_1_download_hdf5):
                    os.makedirs(self.cache_directory, exist_ok=True)
                    cache_hdf5 = \
                        os.path.join(self.cache_directory,
                                     f.replace('.h5', '_cache.h5'))
                    print('*** fast access hdf5:', cache_hdf5)
            else:
                downloaded_file = self.input_file

            print(f'##     Product {product_count}: {basename}'
                  f' (s3 object: {i})')

            if flag_s3_bucket:
                s3_product_path = os.path.join('s3://', bucket_name, path, f)
                vsis3_product_path = s3_product_path.replace('s3://',
                                                             '/vsis3/')

            self.nisar_product_obj = None
            if cache_hdf5 is not None and os.path.isfile(cache_hdf5):
                print(f'        opening cache HDF5 file for fast access:'
                      f' {cache_hdf5}')
                h5_obj = plant.h5py_file_wrapper(cache_hdf5, swmr=True)
                try:
                    self.nisar_product_obj = open_product(cache_hdf5)

                except BaseException:
                    print('WARNING there was an error reading cache file:'
                          f' {cache_hdf5}. Deleting corrupted file.')
                    os.remove(cache_hdf5)

            if self.nisar_product_obj is None or flag_s3_bucket:

                print('*** opening product at s3 path:', s3_product_path)
                h5_obj = plant.h5py_file_wrapper(s3_product_path)

                self.nisar_product_obj = open_product(s3_product_path)

            elif self.nisar_product_obj is None:
                print('*** opening product at:', os.path.join(path, f))
                h5_obj = plant.h5py_file_wrapper(os.path.join(path, f),
                                                 swmr=True)
                self.nisar_product_obj = open_product(os.path.join(path, f))

            current_file_product_type = plant_isce3.get_nisar_product_type(
                h5_obj)
            current_product_level = plant_isce3.get_nisar_product_level(
                h5_obj)
            orbit_direction = plant_isce3.get_nisar_orbit_direction(
                h5_obj)

            if (cache_hdf5 is not None and
                    not os.path.isfile(cache_hdf5) and
                    (self.product_type is None or
                     self.product_type == 'GCOV')):
                print(f'        saving fast access HDF5 file: {cache_hdf5}')

                hdf5_out_obj = self.create_nisar_product_cache(
                    cache_hdf5, h5_obj, current_file_product_type)

                h5_obj.close()
                h5_obj = hdf5_out_obj

                del self.nisar_product_obj
                try:
                    self.nisar_product_obj = open_product(cache_hdf5)
                except BaseException:
                    print('WARNING there was an error creating cache file:'
                          f' {cache_hdf5}')
                    os.remove(cache_hdf5)

            absolute_orbit_number = \
                plant_isce3.get_nisar_product_absolute_orbit_number(h5_obj)
            print('*** absolute_orbit_number:', absolute_orbit_number)
            cycle_number = plant_isce3.get_nisar_product_cycle_number(h5_obj)
            print('*** cycle_number:', cycle_number)
            mission_id = plant_isce3.get_nisar_product_mission_id(h5_obj)
            print('*** mission_id:', mission_id)
            track_number = plant_isce3.get_nisar_product_track_number(h5_obj)
            print('*** track_number:', track_number)
            if self.product_type != 'RRSD':
                frame_number = plant_isce3.get_nisar_product_frame_number(
                    h5_obj)
                if (self.frame_number is not None and
                        frame_number != int(self.frame_number)):
                    continue
                print('*** frame_number:', frame_number)
            else:
                frame_number = None

            is_mixed_mode = bool(
                plant_isce3.get_nisar_product_is_mixed_mode(h5_obj))
            print('*** is_mixed_mode:', is_mixed_mode)

            is_full_frame = bool(
                plant_isce3.get_nisar_product_is_full_frame(h5_obj))
            print('*** is_full_frame:', is_full_frame)

            zero_doppler_start_time = \
                plant_isce3.get_nisar_product_zero_doppler_start_time(h5_obj)
            zero_doppler_end_time = \
                plant_isce3.get_nisar_product_zero_doppler_end_time(h5_obj)
            if current_product_level == 'L2':
                epsg = str(get_product_epsg(h5_obj, current_file_product_type))
            else:
                epsg = None

            if self.product_type is not None:
                if self.product_type != current_file_product_type:
                    continue

            input_rslc_granule = get_input_rslc_granule(
                h5_obj, current_file_product_type)
            print('*** input_rslc_granule:', input_rslc_granule)
            list_of_input_l0b_granules = \
                get_list_of_input_l0b_granules_from_source_data(
                    h5_obj, current_file_product_type)
            print('*** list_of_input_l0b_granules:',
                  list_of_input_l0b_granules)

            kwargs_product_data_to_backscatter = {}
            kwargs_product_data_to_backscatter_str = ''

            if (current_file_product_type == 'GSLC' and not
                    self.rtc_gamma_to_sigma_file and not
                    self.number_of_looks_file and not
                    self.elevation_antenna_pattern_file and not
                    self.noise_equivalent_backscatter_file and not
                    self.doppler_centroid_file):
                kwargs_product_data_to_backscatter['square'] = True
                kwargs_product_data_to_backscatter_str = '--sqrt'

            bounding_polygon_wkt = \
                plant_isce3.get_nisar_product_bounding_polygon(h5_obj)

            bounding_polygon = ogr.CreateGeometryFromWkt(bounding_polygon_wkt)
            min_lon, max_lon, min_lat, max_lat, center_lon, center_lat = \
                self.get_polygon_parameters(bounding_polygon)

            print('*** center_lon:', center_lon)

            if self.bbox:

                outer_polygon_ogr = self.get_bbox_polygon()

                self.print('***        product extents:')
                with plant.PlantIndent():
                    self.print(f'***            min_lat: {min_lat}')
                    self.print(f'***            min_lon: {min_lon}')
                    self.print(f'***            max_lat: {max_lat}')
                    self.print(f'***            max_lon: {max_lon}')

                if not outer_polygon_ogr.Intersects(bounding_polygon):
                    print('Product does not intersect with the selection bbox.'
                          ' Skipping.')
                    continue

            if current_file_product_type == 'STATIC':
                list_of_frequencies_dict = {None: None}
            elif self.frequency is None:
                list_of_frequencies_dict = self.nisar_product_obj.polarizations
            else:
                if (self.frequency
                        not in self.nisar_product_obj.polarizations.keys()):
                    print('WARNING skipping product because frequency'
                          f' {self.frequency} is not present in the product'
                          ' polarizations:',
                          self.nisar_product_obj.polarizations)
                    continue
                list_of_frequencies_dict = {
                    self.frequency:
                    self.nisar_product_obj.polarizations[self.frequency]}

            if self.must_be_quad_pol:
                if ('A' in list_of_frequencies_dict.keys() and
                        set(list_of_frequencies_dict['A']) !=
                        set(['HH', 'HV', 'VH', 'VV'])):
                    print('    skipping non quad-pol frequency A:',
                          list_of_frequencies_dict['A'])
                    continue
                if ('B' in list_of_frequencies_dict.keys() and
                        set(list_of_frequencies_dict['B']) !=
                        set(['HH', 'HV', 'VH', 'VV'])):
                    print('    skipping non quad-pol frequency B:',
                          list_of_frequencies_dict['B'])
                    continue
                print('    quad-pol check passed')

            product_count += 1

            for suffix in ['', '_NATIVE', '_LATLON']:
                self.download_browse(
                    bucket_name, mosaic_kmz_file_list, nisar_product_bucket,
                    path, f, basename, cycle_number, downloaded_file, h5_obj,
                    current_file_product_type, orbit_direction,
                    absolute_orbit_number, mission_id, track_number,
                    frame_number, is_mixed_mode, is_full_frame,
                    zero_doppler_start_time, zero_doppler_end_time,
                    epsg, input_rslc_granule, list_of_input_l0b_granules,
                    center_lon, center_lat, suffix)

            if current_product_level != 'L2':
                epsg = str(4326)

            self.update_tiles_map_dict(tiles_map_by_epsg, bbox_by_epsg,
                                       bounding_polygon, epsg)

            output_dir = self.step_2_directory

            os.makedirs(output_dir, exist_ok=True)
            if os.path.isfile(downloaded_file) and flag_s3_bucket:
                print('        HDF5 already downloaded:', f)

            elif os.path.isfile(downloaded_file):
                print('        HDF5 input file:', f)

            elif self.step_1_download_hdf5:

                os.makedirs(os.path.dirname(downloaded_file),
                            exist_ok=True)
                print('    Step 1: Downloading file:', f)
                try:
                    nisar_product_bucket.download_file(objects.key, f)
                    os.rename(f, downloaded_file)

                except Exception as e:
                    print('WARNING there was an error downloading file:'
                          f' {objects.key}: "{e}"')
                    continue

            else:
                print('        HDF5 not found/downloaded:', f)
                downloaded_file = None

            tec_kwargs = {}
            if self.step_2_download_tec_file:
                print('Downloading TEC file')

                if self.anc_bucket_name is not None:
                    anc_bucket_name = self.anc_bucket_name
                else:
                    anc_bucket_name = bucket_name

                print('Ancillary files bucket name:', anc_bucket_name)

                anc_bucket_name = anc_bucket_name.replace('s3://', '')
                anc_bucket_obj = resource.Bucket(anc_bucket_name)

                start_time_isce3 = \
                    plant_isce3.get_nisar_product_zero_doppler_start_time(
                        h5_obj)
                print('start time:', start_time_isce3)
                end_time_isce3 = \
                    plant_isce3.get_nisar_product_zero_doppler_end_time(
                        h5_obj)
                print('end time:', end_time_isce3)

                start_datetime = \
                    datetime.fromisoformat(
                        isce3.core.DateTime(
                            start_time_isce3).isoformat().split('.')[0])
                end_datetime = \
                    datetime.fromisoformat(
                        isce3.core.DateTime(
                            end_time_isce3).isoformat().split('.')[0])

                tec_count = 0
                flag_success = False
                while not flag_success and tec_count < 24:
                    tec_datetime = \
                        (copy.deepcopy(start_datetime) -
                         timedelta(hours=2 * tec_count))
                    tec_year = tec_datetime.year
                    tec_month = tec_datetime.month
                    tec_day = tec_datetime.day
                    if anc_bucket_name.startswith('nisar-adt'):
                        tec_dir = 'IMAGEN_TOTTEC_ONLY'
                    else:
                        tec_dir = \
                            (f'products/TEC/{tec_year}/{tec_month:02d}/'
                             f'{tec_day:02d}')
                    print('TEC count:', tec_count)
                    print('    TEC directory:', tec_dir)

                    tec_files = plant_isce3.get_files_from_s3_bucket(
                        anc_bucket_obj, tec_dir, extension='.json')

                    print('    len(tec_files) before:', len(tec_files))

                    if len(tec_files) == 0:
                        tec_files = plant_isce3.get_files_from_s3_bucket(
                            anc_bucket_obj, tec_dir, extension='.json',
                            verbose=True)

                    filtered_tec_files = []
                    for tec_file in tec_files:
                        for s in ['.context.json', '.dataset.json',
                                  '.met.json']:
                            if tec_file.endswith(s):
                                break
                        else:
                            filtered_tec_files.append(tec_file)

                    print('    len(filtered_tec_files) after:',
                          len(filtered_tec_files))

                    print('    tec_files:', filtered_tec_files)
                    for tec_file in filtered_tec_files:
                        parsed_tec_filename = parse_tec_filename(tec_file)
                        tec_start_datetime = \
                            parsed_tec_filename['start_datetime']
                        tec_end_datetime = \
                            parsed_tec_filename['end_datetime']
                        print('    NISAR product:', start_datetime,
                              end_datetime)
                        print('    TEC product:', tec_start_datetime,
                              tec_end_datetime)
                        margin = timedelta(seconds=10)
                        flag_success = \
                            (tec_start_datetime < (start_datetime - margin) and
                             tec_end_datetime > (end_datetime + margin))
                        print('    flag success:', flag_success)
                        if flag_success:
                            break
                    tec_count += 1

                if flag_success:

                    tec_basename = os.path.basename(tec_file)

                    if not os.path.isfile(tec_file):
                        print('Downloading TEC file:', tec_basename)
                        anc_bucket_obj.download_file(tec_file, tec_basename)

                    if os.path.isfile(tec_basename):
                        ancillary_files_dir = '1_downloaded_ancillary_data'
                        output_tec_file = os.path.join(ancillary_files_dir,
                                                       tec_basename)

                        os.makedirs(ancillary_files_dir, exist_ok=True)
                        shutil.move(tec_basename, output_tec_file)
                        tec_kwargs['tec_file'] = output_tec_file
                    else:
                        print('WARNING there was an issue downloading the'
                              f' TEC file: {basename}')
                else:
                    print(f'ERROR could not find a TEC file for product {f}')

            h5_obj.close()
            del h5_obj

            output_runconfig = os.path.join(output_dir,
                                            basename + '_gcov.yaml')

            if self.step_2_generate_gcov_runconfig:

                output_l2_basename = f'{basename}.h5'
                output_l2_basename = output_l2_basename.replace('L1', 'L2')
                output_l2_basename = output_l2_basename.replace('RSLC', 'GCOV')
                output_l2 = os.path.join(output_dir,

                                         output_l2_basename)
                geo_kwargs = {}
                if self.plant_geogrid_obj is not None:
                    self.plant_geogrid_obj.print()
                    if plant.isvalid(self.plant_geogrid_obj.step_x):
                        geo_kwargs['step_x'] = self.plant_geogrid_obj.step_x
                    if plant.isvalid(self.plant_geogrid_obj.step_y):
                        geo_kwargs['step_y'] = \
                            self.plant_geogrid_obj.step_y
                    print('geo_kwargs:', geo_kwargs)

                plant_isce3.runconfig(
                    downloaded_file,

                    dem=self.dem_file,

                    output_file=output_runconfig,
                    sas_output_file=output_l2,
                    full_covariance=self.full_covariance,
                    debug=self.flag_debug,
                    force=True,
                    **geo_kwargs,
                    **tec_kwargs)

            for frequency, pols in list_of_frequencies_dict.items():

                if (current_file_product_type != 'STATIC' and
                        frequency not in frequency_epsg_dict.keys()):
                    continue

                if downloaded_file is None and flag_s3_bucket:
                    downloaded_file = vsis3_product_path
                    print('        using remote reference (vsis3):',
                          downloaded_file)

                self.run_process_native_coordinates_freq(
                    kwargs_color, kwargs_product_data_to_backscatter,
                    frequency_epsg_dict, downloaded_file, basename, epsg,
                    output_dir, frequency, pols, current_file_product_type,
                    current_product_level, kwargs_product_data_to_backscatter_str)

        return frequency_epsg_dict, mosaic_kmz_file_list

    def download_browse(
            self, bucket_name, mosaic_kmz_file_list, nisar_product_bucket,
            path, f, basename, cycle_number, downloaded_file, h5_obj,
            current_file_product_type, orbit_direction,
            absolute_orbit_number, mission_id, track_number,
            frame_number, is_mixed_mode, is_full_frame,
            zero_doppler_start_time, zero_doppler_end_time,
            epsg, input_rslc_granule, list_of_input_l0b_granules,
            center_lon, center_lat, suffix):
        png_file = os.path.join(self.step_1_directory,
                                f'{basename}_BROWSE{suffix}.png')

        f_png = f.replace('.h5', f'{suffix}.png')
        png_s3_prefix = os.path.join(path, f_png)

        if (self.step_1_download_browse_png and
                not os.path.isfile(png_file)):

            os.makedirs(os.path.dirname(downloaded_file),
                        exist_ok=True)
            print('    Step 1: Downloading browse file (PNG):', f_png)
            try:
                nisar_product_bucket.download_file(png_s3_prefix, f_png)
            except Exception as e:
                print('WARNING there was an error downloading file:'
                      f' {png_s3_prefix}: "{e}"')

            if os.path.isfile(f_png):
                os.rename(f_png, png_file)

        if os.path.isfile(png_file):
            mosaic_kmz_file_list.append(png_file)

        if suffix == '_NATIVE':
            return

        f_kml = f.replace('.h5', f'{suffix}.kml')
        kml_file = os.path.join(self.step_1_directory,
                                f'{basename}_BROWSE{suffix}.kml')
        kml_s3_prefix = os.path.join(path, f_kml)

        if (self.step_1_download_kml and
                not os.path.isfile(kml_file)):

            os.makedirs(os.path.dirname(downloaded_file),
                        exist_ok=True)
            print('    Step 1: Downloading browse file (KML):', f_kml)
            try:
                nisar_product_bucket.download_file(kml_s3_prefix, f_kml)

            except Exception as e:
                print('WARNING there was an error downloading file:'
                      f' {kml_s3_prefix}: "{e}"')

            if os.path.isfile(f_kml):
                s3_product_path_directory = \
                    os.path.join('s3://', bucket_name, path)

                if (orbit_direction == 'Descending' and
                        epsg is not None and epsg in ['3413', '3031']):
                    icon = 'https://maps.google.com/mapfiles/kml/paddle/D.png'
                elif (orbit_direction == 'Ascending' and
                      epsg is not None and epsg in ['3413', '3031']):
                    icon = 'https://maps.google.com/mapfiles/kml/paddle/A.png'
                elif (orbit_direction == 'Descending' and
                      center_lat < 60):
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-9.png'
                elif (orbit_direction == 'Ascending' and
                      center_lat < 60):
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-15.png'
                elif (orbit_direction == 'Descending' and
                      center_lat < 70):
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-10.png'
                elif (orbit_direction == 'Ascending' and
                      center_lat < 70):
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-14.png'
                elif orbit_direction == 'Descending':
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-11.png'
                elif orbit_direction == 'Ascending':
                    icon = 'https://earth.google.com/images/kml-icons/track-directional/track-13.png'
                else:
                    raise ValueError('Unrecognized orbit pass direction: '
                                     f'"{orbit_direction}"')

                if current_file_product_type in ['GCOV', 'GSLC']:
                    extra_info = ('<p><b>Input RSLC granule:</b>'
                                  f' {input_rslc_granule}</p>')
                    if len(list_of_input_l0b_granules) == 1:
                        extra_info += ('<p><b>Input L0B granule:</b>'
                                       f' {list_of_input_l0b_granules[0]}'
                                       '</p>')
                    else:
                        for i, input_l0b_granule in enumerate(
                                list_of_input_l0b_granules):
                            extra_info += \
                                (f'<p><b>Input L0B granule {i + 1}:</b>'
                                 f' {input_l0b_granule}</p>')
                else:
                    extra_info = ''

                if mission_id == 'NISAR':
                    cycle_number_no_offset = \
                        plant_isce3.get_nisar_product_cycle_number(
                            h5_obj, flag_no_offset=True)
                    extra_cycle = ('<p><b>Cycle number (without offset):'
                                   f'</b> {cycle_number_no_offset}</p>')
                else:
                    extra_cycle = ''

                if epsg is not None:
                    epsg_str = f'<p><b>EPSG code:</b> {epsg}</p>'
                else:
                    epsg_str = ''

                kml_placemark_str = f'''  <Placemark>
        <name></name>
        <description><![CDATA[
            <p><b>Product:</b> {basename}</p>
            <p><b>Product type:</b> {current_file_product_type}</p>{extra_info}
            <p><b>Orbit pass direction:</b> {orbit_direction}</p>{epsg_str}
            <p><b>Absolute orbit number:</b> {absolute_orbit_number}</p>
            <p><b>Cycle number:</b> {cycle_number}</p>{extra_cycle}
            <p><b>Track number:</b> {track_number}</p>
            <p><b>Frame number:</b> {frame_number}</p>
            <p><b>Is mixed mode:</b> {is_mixed_mode}</p>
            <p><b>Is full frame:</b> {is_full_frame}</p>
            <p><b>Zero doppler start time:</b> {zero_doppler_start_time}</p>
            <p><b>Zero doppler end time:</b> {zero_doppler_end_time}</p>
            <p><b>S3 path:</b> {s3_product_path_directory}</p>
            <p><b>Center longitude:</b> {center_lon}</p>
            <p><b>Center latitude:</b> {center_lat}</p>
        ]]></description>
        <Style>
            <IconStyle>
            <Icon>
                <href>{icon}</href>
            </Icon>
            </IconStyle>
        </Style>
        <Point>
            <coordinates>{center_lon},{center_lat}</coordinates>
        </Point>
        </Placemark>
    </Document>
                        '''

                substitute_in_file(
                    f_kml, kml_file,
                    [f'{basename}{suffix}.png', 'overlay image', '</Document>'],
                    [f'{basename}_BROWSE{suffix}.png', basename, kml_placemark_str])

                os.remove(f_kml)

        if os.path.isfile(kml_file):
            mosaic_kmz_file_list.append(kml_file)

    def get_polygon_parameters(self, product_polygon):
        min_lon, max_lon, min_lat, max_lat = \
            product_polygon.GetEnvelope()

        if max_lon - min_lon > 180:

            min_lon, max_lon = [max_lon, min_lon + 360]

        print('*** min_lon:', min_lon)
        print('*** max_lon:', max_lon)

        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        return min_lon, max_lon, min_lat, max_lat, center_lon, center_lat

    def get_bbox_polygon(self):
        bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon = \
            self.bbox

        self.print('***        selection bbox:')
        with plant.PlantIndent():
            self.print(f'*** bbox_min_lat: {bbox_min_lat}')
            self.print(f'*** bbox_min_lon: {bbox_min_lon}')
            self.print(f'*** bbox_max_lat: {bbox_max_lat}')
            self.print(f'*** bbox_max_lon: {bbox_max_lon}')

        outer_ring = ogr.Geometry(ogr.wkbLinearRing)
        outer_ring.AddPoint(bbox_max_lon, bbox_min_lat)
        outer_ring.AddPoint(bbox_max_lon, bbox_max_lat)
        outer_ring.AddPoint(bbox_min_lon, bbox_max_lat)
        outer_ring.AddPoint(bbox_min_lon, bbox_min_lat)
        outer_ring.CloseRings()
        outer_polygon_ogr = ogr.Geometry(ogr.wkbPolygon)
        outer_polygon_ogr.AddGeometry(outer_ring)
        return outer_polygon_ogr

    def create_nisar_product_cache(self, cache_hdf5, h5_obj,
                                   current_file_product_type):
        instrument_name = \
            plant_isce3.get_nisar_product_instrument_name(h5_obj)

        hdf5_out_obj = h5py.File(cache_hdf5, 'a')
        print('*** copying global attributes')
        for key, value in h5_obj.attrs.items():
            hdf5_out_obj.attrs[key] = value

        identification_parent_group_path = f'/science/{instrument_name}'
        identification_path = (f'{identification_parent_group_path}/'
                               'identification')
        metadata_group_path = (f'/science/{instrument_name}/'
                               f'{current_file_product_type}/metadata')
        source_data_processing_information_path = \
            f'{metadata_group_path}/sourceData/processingInformation'
        source_data_parameters_path = \
            (f'{metadata_group_path}/sourceData/processingInformation/'
             'parameters')
        processing_information_path = \
            (f'{metadata_group_path}/processingInformation')

        print('*** identification_path:', identification_path)
        print('*** copying identification group')
        hdf5_out_obj.require_group(identification_parent_group_path)

        h5_obj.copy(identification_path, hdf5_out_obj,
                    name=identification_path)

        if current_file_product_type in ['GCOV', 'GSLC']:
            print('*** copying sourceData parameters group')
            hdf5_out_obj.require_group(source_data_processing_information_path)
            h5_obj.copy(source_data_parameters_path, hdf5_out_obj,
                        name=source_data_parameters_path)

        print('*** requiring metadata group')
        hdf5_out_obj.require_group(processing_information_path)

        inputs_path = f'{processing_information_path}/inputs'
        h5_obj.copy(inputs_path, hdf5_out_obj, name=inputs_path)

        parameters_path = f'{processing_information_path}/parameters'
        runconfig_path = f'{parameters_path}/runConfigurationContents'
        hdf5_out_obj.require_group(parameters_path)
        h5_obj.copy(runconfig_path, hdf5_out_obj, name=runconfig_path)

        print('*** done copying global attributes, identification and metadata'
              ' groups')

        list_of_frequencies_path = \
            (f'/science/{instrument_name}/'
             f'identification/listOfFrequencies')
        list_of_frequencies = h5_obj[list_of_frequencies_path][()]
        print('*** list_of_frequencies:', list_of_frequencies)

        for frequency in list_of_frequencies:
            if not isinstance(frequency, str):
                frequency = frequency.decode()

            print(f'*** frequency: {frequency}')

            if current_file_product_type.startswith('G'):
                frequency_path = \
                    (f'/science/LSAR/{current_file_product_type}/grids/'
                     f'frequency{frequency}')
            else:
                frequency_path = \
                    (f'/science/LSAR/{current_file_product_type}/swaths/'
                     f'frequency{frequency}')

            if current_file_product_type.startswith('G'):
                projection_path = f'{frequency_path}/projection'

                print('*** copying projection group')
                hdf5_out_obj.require_group(frequency_path)
                h5_obj.copy(projection_path, hdf5_out_obj,
                            name=projection_path)

            list_of_polarizations_path = \
                f'{frequency_path}/listOfPolarizations'

            print('*** copying list_of_polarizations group')
            hdf5_out_obj.require_group(frequency_path)
            h5_obj.copy(list_of_polarizations_path, hdf5_out_obj,
                        name=list_of_polarizations_path)
            print('*** done copying projection and list of polarizations'
                  ' groups')

        return hdf5_out_obj

    def run_process_native_coordinates_freq(
            self, kwargs_color, kwargs_product_data_to_backscatter,
            frequency_epsg_dict, downloaded_file, basename, epsg, output_dir,
            frequency, pols, current_product_type, current_product_level,
            kwargs_product_data_to_backscatter_str):

        nlooks_y, nlooks_x = self.get_nlooks(frequency=frequency)

        if current_product_type == 'STATIC':
            frequency_str = ''
        else:
            frequency_str = f'_{frequency}'
        if (self.rtc_gamma_to_sigma_file):
            suffix = (f'{frequency_str}_rtc_gamma_to_sigma')
        elif (self.number_of_looks_file):
            suffix = (f'{frequency_str}_number_of_looks')
        elif (self.elevation_antenna_pattern_file):
            suffix = (f'{frequency_str}_elevation_antenna_pattern')
        elif (self.noise_equivalent_backscatter_file):
            suffix = (f'{frequency_str}_noise_equivalent_backscatter')
        elif (self.doppler_centroid_file):
            suffix = (f'{frequency_str}_doppler_centroid')
        elif nlooks_y != 1 or nlooks_x != 1:
            suffix_ml = f'_ml_{nlooks_y}_{nlooks_x}'
            suffix = (f'{frequency_str}{suffix_ml}')
        else:
            suffix_ml = ''
            suffix = f'{frequency_str}'

        output_file = os.path.join(output_dir, basename + suffix + '.tif')
        output_kmz = os.path.join(output_dir, basename + suffix + '.kmz')
        output_png = os.path.join(output_dir, basename + suffix + '.png')

        if self.step_2_off_diagonal_analysis:
            kwargs_off_diag_analysis = {}
            if self.plot_dataset_name:
                kwargs_off_diag_analysis['plot_dataset_name'] = \
                    self.plot_dataset_name
            if self.plot_date:
                kwargs_off_diag_analysis['plot_date'] = self.plot_date

            output_dir_off_diag_analysis = os.path.join(
                '2_off_diag_analysis', basename + suffix)

            flag_success_off_diag_analysis = False
            error_count = 0
            while not flag_success_off_diag_analysis and error_count < 3:
                try:
                    plant_isce3.off_diagonal_analysis(
                        downloaded_file,
                        output_dir=output_dir_off_diag_analysis,
                        frequency=frequency,
                        output_skip_if_existent=self.output_skip_if_existent,
                        force=True,

                        generate_elevation_profiles=True,
                        remove_cross_multiplication_files=True,
                        nlooks_x=nlooks_x, nlooks_y=nlooks_y,

                        **kwargs_off_diag_analysis)
                    flag_success_off_diag_analysis = True
                except Exception:
                    error_count += 1
                    print('==================================================')
                    print('==================================================')
                    print('==================================================')
                    print('WARNING there was an error during the off-diagonal'
                          ' analysis.')
                    print('==================================================')
                    print('==================================================')
                    print('==================================================')
            if not flag_success_off_diag_analysis:
                print('WARNING the off-diagonal analysis did not complete'
                      ' successfully after 3 attempts. Skipping.')

        if self.step_2_eap_analysis:
            flag_success_eap_analysis = False
            error_count = 0
            while not flag_success_eap_analysis and error_count < 3:
                try:
                    eap_kwargs = {}
                    if self.worldcover:
                        eap_kwargs['worldcover'] = self.worldcover
                    if self.profile_max_in_db is not None:
                        eap_kwargs['profile_max_in_db'] = \
                            self.profile_max_in_db
                    if self.profile_min_in_db is not None:
                        eap_kwargs['profile_min_in_db'] = \
                            self.profile_min_in_db
                    output_dir_eap = os.path.join(
                        '2_eap_analysis', basename + suffix_ml)
                    plant_isce3.rslc_eap_analysis(
                        downloaded_file,
                        frequency=frequency,
                        profiles_directory=output_dir_eap,
                        output_file='""',
                        dem=self.dem_file,
                        output_skip_if_existent=self.output_skip_if_existent,
                        force=True,
                        ignore_noise=True,
                        png_prefix=basename + "_",

                        load_processed_backscatter_image=True,
                        save_multilooked_backscatter_image=True,
                        save_processed_backscatter_image=True,
                        save_multilooked_backscatter_png=True,
                        save_processed_backscatter_png=True,
                        save_profile_plot_png=True,
                        create_plots_with_predefined_thresholds=self.flag_create_plots_with_predefined_thresholds,
                        nlooks_x=nlooks_x,
                        nlooks_y=nlooks_y,
                        **eap_kwargs)
                    flag_success_eap_analysis = True
                except Exception:
                    error_count += 1
                    print('==================================================')
                    print('==================================================')
                    print('==================================================')
                    print('WARNING: There was an error during the EAP'
                          ' analysis.')

                    print('==================================================')
                    print('==================================================')
                    print('==================================================')
            if not flag_success_eap_analysis:
                print('WARNING The EAP analysis did not complete successfully'
                      ' after 3 attempts. Skipping.')

        masked_data_kwargs = {}

        if self.masked_data_file:
            masked_data_kwargs['masked_data_file'] = True
            masked_data_kwargs_str = '--masked-data'
        elif self.rtc_gamma_to_sigma_file:
            masked_data_kwargs['rtc_gamma_to_sigma_file'] = True
            masked_data_kwargs_str = '--rtc-gamma-to-sigma'
        elif self.number_of_looks_file:
            masked_data_kwargs['number_of_looks_file'] = True
            masked_data_kwargs_str = '--number-of-looks'
        elif self.elevation_antenna_pattern_file:
            masked_data_kwargs['elevation_antenna_pattern_file'] = True
            masked_data_kwargs_str = '--elevation-antenna-pattern'
        elif self.noise_equivalent_backscatter_file:
            masked_data_kwargs['noise_equivalent_backscatter_file'] = True
            masked_data_kwargs_str = '--noise-equivalent-backscatter'
        elif self.doppler_centroid_file:
            masked_data_kwargs['doppler_centroid_file'] = True
            masked_data_kwargs_str = '--doppler-centroid'
        else:
            masked_data_kwargs['data_file'] = True
            masked_data_kwargs_str = '--data'

        bands_rgb_kwargs = {}

        if current_product_type != 'STATIC':
            nbands = len(pols)
            if nbands == 4:

                bands_rgb_kwargs['bands'] = '0,1,2'
        else:
            nbands = 1
            pols = [None]

        input_ref = f'NISAR:{downloaded_file}:{frequency}'
        if (self.step_2_generate_cog_rgb and not os.path.isfile(output_file)):

            plant_isce3.util(
                input_ref,

                nlooks_x=nlooks_x,
                nlooks_y=nlooks_y,
                output_file=output_file,
                force=True,
                **masked_data_kwargs,
                **bands_rgb_kwargs,

                **kwargs_product_data_to_backscatter)

        start_time_all_pols = time.time()

        processes = []

        for band, pol in enumerate(pols):
            if self.pol_list is not None and pol not in self.pol_list:
                print(f'***        skipping polarization {pol} based on user'
                      ' input')
                continue
            if pol is not None:
                output_file_pol = \
                    os.path.join(output_dir, f'{basename}{suffix}_{pol}.tif')
            else:
                output_file_pol = \
                    os.path.join(output_dir, f'{basename}{suffix}.tif')

            if (self.step_2_generate_cog_parallel and
                    not os.path.isfile(output_file_pol)):

                input_ref = f'NISAR:{downloaded_file}:{frequency}'
                command = \
                    (f'python3 {self.plant_isce3_util_path} {input_ref}'
                     f' --nlooks-x {nlooks_x}'
                     f' --nlooks-y {nlooks_y} --output-file {output_file_pol}'
                     f' --force --band {band} {masked_data_kwargs_str}'
                     f' {kwargs_product_data_to_backscatter_str}')

                p = subprocess.Popen(command, shell=True)
                processes.append(p)

            elif (self.step_2_generate_cog and
                    not os.path.isfile(output_file_pol)):
                start_time = time.time()
                input_ref = f'NISAR:{downloaded_file}:{frequency}'
                plant_isce3.util(
                    input_ref,

                    nlooks_x=nlooks_x,
                    nlooks_y=nlooks_y,
                    output_file=output_file_pol,
                    force=True,
                    band=band,
                    **masked_data_kwargs,

                    **kwargs_product_data_to_backscatter)

                end_time = time.time()
                if pol is not None:
                    print(f'        time to generate COG for pol {pol}:'
                          f' {end_time - start_time:.2f} seconds')

        if self.step_2_generate_cog_parallel and len(processes) > 0:

            for p in processes:
                if p.wait() != 0:
                    print('ERROR A COG generation process did not finish'
                          ' successfully')

            print("All COG generation processes finished")
            end_time_all_pols = time.time()
            print(f'        time to generate COG for pol {pol}:'
                  f' {end_time_all_pols - start_time_all_pols:.2f} seconds')

        for band, pol in enumerate(pols):
            if self.pol_list is not None and pol not in self.pol_list:
                print(f'***        skipping polarization {pol} based on user'
                      ' input')
                continue

            output_file_pol = \
                os.path.join(output_dir, f'{basename}{suffix}_{pol}.tif')

            if os.path.isfile(output_file_pol):
                if pol not in frequency_epsg_dict[frequency].keys():
                    frequency_epsg_dict[frequency][pol] = {
                        epsg: [output_file_pol]
                    }
                elif epsg not in frequency_epsg_dict[frequency][pol].keys():
                    frequency_epsg_dict[frequency][pol][epsg] = \
                        [output_file_pol]
                else:
                    frequency_epsg_dict[frequency][pol][epsg].append(
                        output_file_pol)

        if self.step_2_generate_cog:
            end_time_all_pols = time.time()
            print(f'        time to generate COGs for all polarizations:'
                  f' {end_time_all_pols - start_time_all_pols:.2f} seconds')

        if (self.step_2_generate_kmz and not os.path.isfile(output_kmz) and
                os.path.isfile(output_file) and current_product_level == 'L2'):
            self.util(output_file, output_file=output_kmz, force=True)

        elif (self.step_2_generate_kmz and not os.path.isfile(output_kmz) and
              current_product_level == 'L2'):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            plant_isce3.util(
                input_ref,

                nlooks_x=nlooks_x,
                nlooks_y=nlooks_y,
                output_file=output_kmz, force=True,

                **masked_data_kwargs,
                **bands_rgb_kwargs,
                **kwargs_product_data_to_backscatter,
                **kwargs_color)
        elif self.step_2_generate_kmz and not os.path.isfile(output_kmz):
            if os.path.isfile(output_file):
                rslc_file = output_file
            else:
                rslc_file = downloaded_file

            plant_isce3.geocode(rslc_file,

                                dem_file=self.dem_file,
                                output_file=output_kmz,
                                force=True,

                                **bands_rgb_kwargs,
                                **kwargs_color)

        if (self.step_2_generate_png and not os.path.isfile(output_png) and
                os.path.isfile(output_file)):
            self.util(output_file, output_file=output_png, force=True)
        elif (self.step_2_generate_png and not os.path.isfile(output_png)):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            plant_isce3.util(input_ref,

                             nlooks_x=nlooks_x,
                             nlooks_y=nlooks_y,
                             cmap_max=self.cmap_max,
                             cmap_min=self.cmap_min,
                             output_file=output_png,
                             force=True,
                             prefix=f'{basename}_',
                             **masked_data_kwargs,

                             **kwargs_product_data_to_backscatter,
                             **bands_rgb_kwargs,
                             **kwargs_color)

        if (self.step_2_generate_all_layers_kmz and
                current_product_level == 'L2'):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            plant_isce3.util(input_ref,
                             output_dir=os.path.join(output_dir, basename),

                             nlooks_x=nlooks_x,
                             nlooks_y=nlooks_y,
                             output_file=output_kmz,
                             all_layers=True,
                             prefix=f'{basename}_',
                             ext='.kmz',
                             force=True

                             )

        if self.step_2_generate_all_layers_png:
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            plant_isce3.util(input_ref,
                             output_dir=os.path.join(output_dir, basename),

                             nlooks_x=nlooks_x,
                             nlooks_y=nlooks_y,

                             output_file=output_png,
                             all_layers=True,
                             ext='.png',
                             force=True,

                             )

    def create_tiles(self, flag_generate_tiles, flag_generate_tiles_parallel,
                     flag_generate_tiles_kmz,
                     flag_generate_tiles_rgb_kmz,
                     flag_generate_tiles_ab_kmz, flag_vrts,
                     frequency, orbit_direction_str,
                     min_lat, max_lat, min_lon, max_lon,
                     tiles_map, flag_last_pol, suffix_list, suffix_rgb,

                     list_of_epsg_vrts,

                     output_dir_prefix,
                     suffix=''):

        min_lat = int(np.floor(min_lat))
        max_lat = int(np.ceil(max_lat))
        min_lon = int(np.floor(min_lon))
        max_lon = int(np.ceil(max_lon))

        if self.bbox:
            bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon = \
                self.bbox
            min_lat = max(min_lat, int(np.floor(bbox_min_lat)))
            max_lat = min(max_lat, int(np.ceil(bbox_max_lat)))
            min_lon = max(min_lon, int(np.floor(bbox_min_lon)))
            max_lon = min(max_lon, int(np.ceil(bbox_max_lon)))

        print('Extents:')
        print('    min lon:', min_lon)
        print('    max lon:', max_lon)
        print('    min lat:', min_lat)
        print('    max lat:', max_lat)

        file_list = []

        for lat in range(min_lat, max_lat + 1):
            sn_str = 'S' if lat < 0 else 'N'
            lat_str = f'{sn_str}{abs(lat):02d}_00'
            processes = []

            for lon in range(min_lon, max_lon + 1):
                we_str = 'W' if lon < 0 else 'E'
                lon_str = f'{we_str}{abs(lon):03d}_00'

                if not self.check_process_tile(tiles_map, lat, lon):
                    continue

                tile_file = os.path.join(
                    f'{output_dir_prefix}',
                    f'{self.output_files_prefix}{self.product_type}_{lat_str}'
                    f'_{lon_str}{suffix}{self.output_files_suffix}.tif')

                if os.path.isfile(tile_file):
                    file_list.append(tile_file)
                    continue

                if (self.plant_geogrid_obj is not None and
                        self.plant_geogrid_obj.step_x is not None):
                    step_x = self.plant_geogrid_obj.step_x
                else:
                    step_x = res_deg_dict[frequency]
                if (self.plant_geogrid_obj is not None and
                        self.plant_geogrid_obj.step_y is not None):
                    step_y = self.plant_geogrid_obj.step_y
                else:
                    step_y = res_deg_dict[frequency]

                if flag_generate_tiles_parallel:
                    command = \
                        (f'python3 {self.plant_mosaic_path}'
                         f' {" ".join(list_of_epsg_vrts)}'
                         f' --output-file {tile_file}'
                         f' --bbox {lat} {lat + 1} {lon} {lon + 1}'
                         f' --step-x {step_x} --step-y {step_y}'
                         f' --force --in-null nan --out-null nan'
                         f' --of "{cog_str}" --log-enabled'
                         f' --interp average --out-projection wgs84')

                    p = subprocess.Popen(command, shell=True)
                    processes.append((p, tile_file))
                    while len(processes) >= self.n_parallel_processes:
                        for p, tile_file in processes[:]:
                            if p.poll() is None:
                                continue

                            if os.path.isfile(tile_file):
                                file_list.append(tile_file)

                            processes.remove((p, tile_file))
                        time.sleep(0.001)
                    continue

                elif flag_generate_tiles:

                    try:
                        plant.mosaic(*list_of_epsg_vrts,
                                     output_file=tile_file,
                                     bbox=[lat, lat + 1, lon, lon + 1],
                                     step_x=step_x,
                                     step_y=step_y,
                                     force=True, in_null='nan',
                                     out_null='nan', of=cog_str,
                                     interp='average',
                                     out_projection='wgs84')

                        file_list.append(tile_file)
                    except Exception:
                        error_message = plant.get_error_message()
                        print(error_message.replace('ERROR', 'WARNING'))

            if flag_generate_tiles_parallel and len(processes) > 0:

                for p, tile_file in processes:
                    if p.wait() != 0:
                        print('WARNING A tile generation process did not'
                              ' finish successfully')
                        continue
                    if os.path.isfile(tile_file):
                        file_list.append(tile_file)

        for lat in range(min_lat, max_lat + 1):
            sn_str = 'S' if lat < 0 else 'N'
            lat_str = f'{sn_str}{abs(lat):02d}_00'

            for lon in range(min_lon, max_lon + 1):
                we_str = 'W' if lon < 0 else 'E'
                lon_str = f'{we_str}{abs(lon):03d}_00'

                if not self.check_process_tile(tiles_map, lat, lon):
                    continue

                tile_file = os.path.join(
                    f'{output_dir_prefix}',
                    f'{self.output_files_prefix}{self.product_type}_{lat_str}'
                    f'_{lon_str}{suffix}{self.output_files_suffix}.tif')

                if not os.path.isfile(tile_file):
                    continue

                tile_kmz_file = os.path.join(
                    f'{output_dir_prefix}_tiles_kmz',
                    f'{self.output_files_prefix}{self.product_type}_{lat_str}'
                    f'_{lon_str}{suffix}{self.output_files_suffix}.kmz')

                tile_rgb_kmz_file = os.path.join(
                    f'{output_dir_prefix}_tiles_kmz',
                    f'{self.output_files_prefix}{self.product_type}_{lat_str}'
                    f'_{lon_str}{suffix_rgb}{self.output_files_suffix}.kmz')

                tile_kmz_ab_hh_file = os.path.join(
                    f'{output_dir_prefix}_tiles_ab',
                    f'{self.output_files_prefix}{self.product_type}_{lat_str}'
                    f'_{lon_str}_AB'
                    f'_HH{orbit_direction_str}{self.output_files_suffix}'
                    '.kmz')

                if (flag_generate_tiles_kmz and

                        not os.path.isfile(tile_kmz_file)):

                    self.util(tile_file, output_file=tile_kmz_file,

                              force=True)

                if (flag_generate_tiles_rgb_kmz and flag_last_pol and
                        not os.path.isfile(tile_rgb_kmz_file)):

                    rgb_file_list = []

                    for current_suffix in suffix_list:
                        current_tile_file = os.path.join(
                            f'{output_dir_prefix}',
                            f'{self.product_type}_{lat_str}_{lon_str}'
                            f'{current_suffix}.tif')
                        if not os.path.isfile(current_tile_file):
                            continue
                        rgb_file_list.append(current_tile_file)

                    self.util(*rgb_file_list,
                              output_file=tile_rgb_kmz_file,

                              force=True)

                if (flag_generate_tiles_ab_kmz and
                        os.path.isfile(tile_file) and
                        frequency == 'B' and
                        not os.path.isfile(tile_kmz_ab_hh_file)):

                    self.util(
                        tile_file.replace(
                            '_B_', '_A_'), tile_file.replace(
                            'A', 'B'), output_file=tile_kmz_ab_hh_file, band=0, force=True)

        if len(file_list) == 0:
            return

        vrt_file = (
            f'{output_dir_prefix}/{self.output_files_prefix}mosaic{suffix}'
            f'{self.output_files_suffix}.vrt'
        )
        if flag_vrts:
            os.makedirs(output_dir_prefix, exist_ok=True)

            if os.path.isfile(vrt_file):
                os.remove(vrt_file)
            gdal.BuildVRT(vrt_file, file_list, srcNodata='nan',
                          VRTNodata='nan',
                          resampleAlg='average',
                          outputBounds=[min_lon, min_lat, max_lon, max_lat])

            print('##        file saved:', vrt_file)
            add_overviews_vrt(vrt_file)
            print(f'        file updated: {vrt_file} (added overviews)')

        return vrt_file

    def check_process_tile(self, tiles_map, lat, lon):
        min_lat = lat
        max_lat = lat + 1
        min_lon = lon
        max_lon = lon + 1
        lat_index_beg = 180 - (int(np.ceil(max_lat)) + 90)
        lat_index_end = 180 - (int(np.floor(min_lat)) + 90) + 1
        lon_index_beg = int(np.floor(min_lon)) + 180
        lon_index_end = int(np.ceil(max_lon)) + 180 + 1
        flag_process = tiles_map[lat_index_beg:lat_index_end,
                                 lon_index_beg:lon_index_end].any()

        return flag_process

    def update_tiles_map_dict(self, tiles_map_by_epsg,
                              bbox_by_epsg, polygon_geometry, epsg):
        if epsg not in tiles_map_by_epsg.keys():

            epsg_tiles_map = np.zeros((180, 360), dtype=np.byte)
            epsg_min_lat = +90
            epsg_max_lat = -90
            epsg_min_lon = +180
            epsg_max_lon = -180
        else:

            epsg_tiles_map = tiles_map_by_epsg[epsg]
            epsg_min_lon, epsg_max_lon, epsg_min_lat, epsg_max_lat = \
                bbox_by_epsg[epsg]

        min_lon, max_lon, min_lat, max_lat = polygon_geometry.GetEnvelope()
        print('***    min_lon, max_lon, min_lat, max_lat:',
              min_lon, max_lon, min_lat, max_lat)

        epsg_min_lon = min([epsg_min_lon, min_lon])
        epsg_max_lon = max([epsg_max_lon, max_lon])
        epsg_min_lat = min([epsg_min_lat, min_lat])
        epsg_max_lat = max([epsg_max_lat, max_lat])

        lat_index_beg = 180 - (int(np.ceil(max_lat)) + 90)
        lat_index_end = 180 - (int(np.floor(min_lat)) + 90) + 1
        lon_index_beg = int(np.floor(min_lon)) + 180
        lon_index_end = int(np.ceil(max_lon)) + 180 + 1
        epsg_tiles_map[lat_index_beg:lat_index_end,
                       lon_index_beg:lon_index_end] = 1

        tiles_map_by_epsg[epsg] = epsg_tiles_map
        bbox_by_epsg[epsg] = \
            epsg_min_lon, epsg_max_lon, epsg_min_lat, epsg_max_lat

        if 'mosaic' in tiles_map_by_epsg.keys():
            mosaic_tiles_map = tiles_map_by_epsg['mosaic']
        else:
            mosaic_tiles_map = np.zeros((180, 360), dtype=np.uint16)

        mosaic_tiles_map = mosaic_tiles_map + tiles_map_by_epsg[epsg]
        tiles_map_by_epsg['mosaic'] = mosaic_tiles_map

        if 'mosaic' in bbox_by_epsg.keys():
            mosaic_min_lon, mosaic_max_lon, mosaic_min_lat, mosaic_max_lat = \
                bbox_by_epsg['mosaic']
        else:
            mosaic_min_lat = +90
            mosaic_max_lat = -90
            mosaic_min_lon = +180
            mosaic_max_lon = -180

        mosaic_min_lon = min([mosaic_min_lon, epsg_min_lon])
        mosaic_max_lon = max([mosaic_max_lon, epsg_max_lon])
        mosaic_min_lat = min([mosaic_min_lat, epsg_min_lat])
        mosaic_max_lat = max([mosaic_max_lat, epsg_max_lat])

        if self.bbox:
            bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon = \
                self.bbox
            mosaic_min_lat = max(mosaic_min_lat, int(np.floor(bbox_min_lat)))
            mosaic_max_lat = min(mosaic_max_lat, int(np.ceil(bbox_max_lat)))
            mosaic_min_lon = max(mosaic_min_lon, int(np.floor(bbox_min_lon)))
            mosaic_max_lon = min(mosaic_max_lon, int(np.ceil(bbox_max_lon)))

        bbox_by_epsg['mosaic'] = \
            mosaic_min_lon, mosaic_max_lon, mosaic_min_lat, mosaic_max_lat


def substitute_in_file(filename, output_file, old_substring_list,
                       new_substring_list):

    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    if isinstance(old_substring_list, str):
        old_substring_list = [old_substring_list]
    if isinstance(new_substring_list, str):
        new_substring_list = [new_substring_list]

    for old_substring, new_substring in zip(old_substring_list,
                                            new_substring_list):

        content = content.replace(old_substring, new_substring)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(content)


def load_aws_credentials_boto3(profile="default"):
    session = boto3.Session(profile_name=profile)
    creds = session.get_credentials()
    if creds is None:
        return
    frozen_creds = creds.get_frozen_credentials()
    driver_kwds = {
        "aws_region": b"us-west-2",
        "secret_id": frozen_creds.access_key.encode(),
        "secret_key": frozen_creds.secret_key.encode(),
    }
    if frozen_creds.token:
        driver_kwds["session_token"] = frozen_creds.token.encode()
    return driver_kwds


def get_input_rslc_granule(h5_obj, product_type):

    if product_type not in ['GCOV', 'GSLC']:
        return

    input_rslc_granule = \
        h5_obj[f'/science/LSAR/{product_type}/metadata/'
               'processingInformation/inputs/l1SlcGranules'][()]
    input_rslc_granule = [d.decode() for d in input_rslc_granule]
    if len(input_rslc_granule) == 1:
        input_rslc_granule = input_rslc_granule[0]
    return input_rslc_granule


def get_list_of_input_l0b_granules_from_source_data(
        h5_obj, product_type, flag_basename=True):

    if product_type not in ['GCOV', 'GSLC']:
        return

    runconfig_contents = \
        h5_obj[f'/science/LSAR/{product_type}/metadata/'
               'sourceData/processingInformation/parameters/'
               'runConfigurationContents'][()]
    if not isinstance(runconfig_contents, str):
        runconfig_contents = runconfig_contents.decode()

    runconfig_contents_lines = runconfig_contents.split('\n')

    flag_into_input_file_path = False
    list_of_l0b_granules = []
    for line in runconfig_contents_lines:
        if 'input_file_path' in line:
            flag_into_input_file_path = True
            continue
        if flag_into_input_file_path and 'RRSD' in line:
            line = line.replace('-', '')
            input_l0b_granule = line.strip()
            if flag_basename:
                input_l0b_granule = os.path.basename(
                    input_l0b_granule)
            list_of_l0b_granules.append(input_l0b_granule)
        else:
            flag_into_input_file_path = False

    return list_of_l0b_granules


def get_product_epsg(h5_obj, product_type):

    if product_type is None or product_type != 'STATIC':
        list_of_frequencies = h5_obj['/science/LSAR/identification/'
                                     'listOfFrequencies']
        first_frequency = list_of_frequencies[0].decode()
        projection = h5_obj[f'/science/LSAR/{product_type}/grids/'
                            f'frequency{first_frequency}/projection']
    else:
        projection = h5_obj[f'/science/LSAR/STATIC/grids/projection']
    epsg_code = projection.attrs['epsg_code']
    return epsg_code


def add_overviews_vrt(vrt_file):
    command = ('gdaladdo '
               ' --config VRT_VIRTUAL_OVERVIEWS YES'

               f' {vrt_file} 2 4 8 16 32 64 128')
    plant.execute(command)


def add_overviews_tif(tif_file):
    command = ('gdaladdo '
               ' --config VRT_VIRTUAL_OVERVIEWS YES'
               ' -r average'

               f' {tif_file} 2 4 8 16 32 64 128')
    plant.execute(command)


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        with PlantIsce3BatchProcessing(parser, argv) as self_obj:
            ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
