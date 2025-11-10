#!/usr/bin/env python3

import os
import sys
import plant
import plant_isce3

import numpy as np
import isce3
from osgeo import gdal, ogr

import h5py
import boto3
import glob
import pickle

res_deg_dict = {'A': 1.0 / 3600,
                'B': 1.0 / 3600}

filter_method = plant_isce3.filter

cog_str = 'COG # resampling_algorithm=average # overviews_list=2,4,8,16,32,64,128'


def get_parser():

    descr = ''
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=2,
                            dem_file=1,
                            default_output_options=1,
                            default_flags=1,
                            output_format=1,
                            multilook=1,
                            output_dir=1,
                            bbox=1,
                            separate=1,
                            output_file=1)

    plant_isce3.add_arguments(parser,
                              frequency=1)

    parser.add_argument('--product-type',
                        type=str,
                        dest='product_type',
                        help='Product type')

    parser.add_argument('--filename-must-include',
                        type=str,
                        nargs='*',
                        dest='filename_must_include',
                        help='List of strings that the input products should'
                        ' include (at least one)')

    parser.add_argument('--filename-must-not-include',
                        type=str,
                        nargs='*',
                        dest='filename_must_not_include',
                        help=('List of strings that the input products should'
                              ' not include'))

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
                        dest='step_1_download_browse',
                        help='Download browse images (PNG files)')

    parser.add_argument('--step-1-download-kml',
                        action='store_true',
                        dest='step_1_download_kml',
                        help='Download browse KML files')

    parser.add_argument('--step-2-generate-cog',
                        action='store_true',
                        dest='step_2_generate_cog',
                        help='Generate Cloud-Optimized GeoTIFFs (COGs)')

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

    parser.add_argument('--step-3-generate-vrt',
                        action='store_true',
                        dest='step_3_generate_vrt',
                        help='Generate VRT projection files')

    parser.add_argument('--step-4-generate-tiles',
                        action='store_true',
                        dest='step_4_generate_tiles_tiles',
                        help='Generate KMZ files')

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
                        help='Generate mosaic KMZ files of a single-'
                        'polarization')

    parser.add_argument('--step-7-generate-mosaic-kmz',
                        action='store_true',
                        dest='step_7_generate_mosaic_kmz',
                        help='Generate mosaic KMZ file')

    parser.add_argument('--step-8-generate-tile-map-kmz',
                        action='store_true',
                        dest='step_8_generate_time_map_kmz',
                        help='Generate tile map KMZ file')

    parser.add_argument('--nlooks-x-freq-a',
                        '--nlooks-x-a',
                        type=int,
                        help=('Number of looks in the X direction'
                              ' for frequency A (when available)'),
                        dest='nlooks_x_a')

    parser.add_argument('--nlooks-y-freq-a',
                        '--nlooks-y-a',
                        type=int,
                        help=('Number of looks in the X direction'
                              ' for frequency A (when available)'),
                        dest='nlooks_y_a')

    parser.add_argument('--nlooks-x-freq-b',
                        '--nlooks-x-b',
                        type=int,
                        help=('Number of looks in the X direction'
                              ' for frequency B (when available)'),
                        dest='nlooks_x_b')

    parser.add_argument('--nlooks-y-freq-b',
                        '--nlooks-y-b',
                        type=int,
                        help=('Number of looks in the X direction'
                              ' for frequency B (when available)'),
                        dest='nlooks_y_b')

    parser.add_argument('--max-number-products',
                        type=int,
                        dest='max_number_products',
                        help='Maximum number of products to process')

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

    return parser


class PlantIsce3BatchProcessing(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        self.replace_null = False
        super().__init__(parser, argv)

    def run(self):

        self.print(f'input file: {self.input_file}')

        aws_credentials_file = f"{os.path.expanduser('~')}/.aws/credentials"
        gdal.SetConfigOption('AWS_CONFIG_FILE', aws_credentials_file)

        input_file_splitted = self.input_file.split('/')

        flag_s3_bucket = input_file_splitted[0] == 's3:'

        self.print(f'flag s3 bucket: {flag_s3_bucket}')

        if (flag_s3_bucket and len(input_file_splitted) > 1 and
                input_file_splitted[1] != ''):
            self.print(f'ERROR invalid s3 path: {input_file_splitted}')
            return

        if flag_s3_bucket:
            bucket_name = input_file_splitted[2]
            self.print(f's3 bucket: {bucket_name}')
            s3_prefix = '/'.join(input_file_splitted[3:])
            self.print(f's3 prefix: {s3_prefix}')

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
            'pickle_files/frequency_epsg_dict.pkl'
        tiles_map_by_epsg_pickle_file = 'pickle_files/tiles_map_by_epsg.pkl'
        bbox_by_epsg_pickle_file = 'pickle_files/bbox_by_epsg.pkl'

        if self.skip_step_1_and_load_cogs_from:
            search_pattern = os.path.join(
                self.skip_step_1_and_load_cogs_from, '**', '*.tif')
            file_list = glob.glob(search_pattern,
                                  recursive=True)

            frequency_epsg_dict = {}

            for tif_file in file_list:
                print(f'*** evaluating file {tif_file}')
                image_obj = plant.read_image(tif_file)
                metadata = image_obj.metadata

                frequency = None
                pol = None
                for key, value in metadata.items():
                    print(f"{key}: {value}")
                    if key == 'FREQUENCY':
                        frequency = value
                        print('frequency:', frequency)
                        continue
                    if key == 'POLARIZATION':
                        pol = value
                        print('polarization:', pol)
                    if key == 'BOUNDING_POLYGON':
                        bounding_polygon = value
                        print('bounding polygon:', bounding_polygon)

                if frequency is None or pol is None:
                    print(f'Unrecognized file: {tif_file}. Skipping.')
                    continue

                epsg = image_obj.geogrid.epsg
                print('epsg:', epsg)

                update_tiles_map_dict(tiles_map_by_epsg, bbox_by_epsg,
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

                print('frequency_epsg_dict:', frequency_epsg_dict)

        else:
            frequency_epsg_dict = self.step_1_2_processing_native_coordinates(
                flag_s3_bucket, bucket_name, s3_prefix, kwargs_color,
                tiles_map_by_epsg, bbox_by_epsg)

        if self.step_1_save_pickle_files:
            os.makedirs('pickle_files', exist_ok=True)
            with open(frequency_epsg_dict_pickle_file, 'wb') as pickle_file:
                pickle.dump(frequency_epsg_dict, pickle_file)
            print('file saved:', frequency_epsg_dict_pickle_file)

            with open(tiles_map_by_epsg_pickle_file, 'wb') as pickle_file:
                pickle.dump(tiles_map_by_epsg, pickle_file)
            print('file saved:', tiles_map_by_epsg_pickle_file)

            with open(bbox_by_epsg_pickle_file, 'wb') as pickle_file:
                pickle.dump(bbox_by_epsg, pickle_file)
            print('file saved:', bbox_by_epsg_pickle_file)

        print('## Processing steps 3-5')

        orbit_pass_direction_str = ''

        for frequency, pol_dict in frequency_epsg_dict.items():

            suffix_list = []

            for pol, epsg_dict in pol_dict.items():

                suffix = f'_{frequency}_{pol}{orbit_pass_direction_str}'
                suffix_rgb = f'_{frequency}{orbit_pass_direction_str}'

                suffix_list.append(suffix)

                for epsg, file_list in epsg_dict.items():

                    list_of_output_files = []

                    print(f'## Processing EPSG: {epsg} ({len(file_list)})')

                    vrt_file = (f'{self.step_3_directory}/EPSG{epsg}{suffix}'
                                f'{orbit_pass_direction_str}.vrt')

                    if (self.step_3_generate_vrt and
                            not os.path.isfile(vrt_file)):
                        os.makedirs(self.step_3_directory, exist_ok=True)
                        print('    Step 3: Building VRT from files:',
                              file_list, ', output file:', vrt_file)
                        if os.path.isfile(vrt_file):
                            os.remove(vrt_file)

                        gdal.BuildVRT(vrt_file, file_list, srcNodata='nan',
                                      VRTNodata='nan')
                        print('        file saved:', vrt_file)
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

                    if vrt_file is None:
                        continue

                    list_of_output_files.append(vrt_file)

                flag_last_pol = pol == list(pol_dict.keys())[-1]

                vrt_file = self.run_processing_geographic(
                    tiles_map_by_epsg, bbox_by_epsg, orbit_pass_direction_str,
                    frequency, flag_last_pol, suffix_list, suffix_rgb,
                    suffix, list_of_output_files)

            if self.step_7_generate_mosaic_kmz:

                mosaic_a_vrt_file = (f'{self.step_5_directory}/mosaic_A_{pol}'
                                     f'{orbit_pass_direction_str}.vrt')
                mosaic_b_vrt_file = (f'{self.step_5_directory}/mosaic_B_{pol}'
                                     f'{orbit_pass_direction_str}.vrt')

                print('    Step 6: KMZ')
                kmz_file = (f'{self.step_5_directory}/mosaic_AB_{pol}'
                            f'{orbit_pass_direction_str}.kmz')

                print('pol_vrt_list:', vrt_file)
                self.util(mosaic_a_vrt_file, mosaic_b_vrt_file,
                          output_file=kmz_file, force=True,

                          in_null=np.nan)

        print('done')

    def run_processing_geographic(
            self, tiles_map_by_epsg, bbox_by_epsg, orbit_pass_direction_str,
            frequency, flag_last_pol, suffix_list, suffix_rgb,
            suffix, list_of_output_files):
        mosaic_tiles_map = tiles_map_by_epsg['mosaic']
        mosaic_min_lon, mosaic_max_lon, mosaic_min_lat, mosaic_max_lat = \
            bbox_by_epsg['mosaic']

        vrt_file = self.create_tiles(
            self.step_4_generate_tiles_tiles,
            self.step_4_generate_tiles_kmz,
            self.step_4_generate_tiles_rgb_kmz,
            self.step_4_generate_tiles_ab_kmz,
            self.step_5_generate_mosaic_vrt,
            frequency, orbit_pass_direction_str,
            mosaic_min_lat, mosaic_max_lat,
            mosaic_min_lon, mosaic_max_lon,
            mosaic_tiles_map,
            flag_last_pol, suffix_list, suffix_rgb,

            list_of_output_files,

            output_dir_prefix=self.step_4_directory,
            suffix=suffix)

        if self.step_6_generate_mosaic_kmz:
            print('    Step 6: Kmz')
            kmz_file = f'{self.step_5_directory}/mosaic{suffix}.kmz'
            self.util(vrt_file, output_file=kmz_file, force=True,

                      in_null=np.nan)

        if self.step_6_generate_mosaic_pol_kmz:
            for pol_count, pol in enumerate(['_HH', '_HV']):
                mosaic_vrt_file = \
                    f'{self.step_5_directory}/mosaic{suffix}.vrt'
                print('    Step 6: Kmz')
                kmz_file = f'{self.step_5_directory}/mosaic{suffix}.kmz'
                print('vrt_file:', vrt_file)
                self.util(mosaic_vrt_file, output_file=kmz_file,
                          force=True,

                          in_null=np.nan)

        if self.step_8_generate_time_map_kmz:
            tiles_map_geotransform = [-180, 1, 0, 90, 0, -1]

            plant.save_image(
                mosaic_tiles_map.copy(),
                f'{self.step_5_directory}/tiles_map{suffix}.kmz',
                geotransform=tiles_map_geotransform,
                force=True)
            for epsg, tile_map in tiles_map_by_epsg.items():
                if epsg == 'mosaic':
                    continue
                plant.save_image(
                    tile_map.copy(),
                    f'{self.step_5_directory}/tiles_map_{epsg}_{suffix}.kmz',
                    geotransform=tiles_map_geotransform,
                    force=True)

        return vrt_file

    def step_1_2_processing_native_coordinates(
            self, flag_s3_bucket, bucket_name, s3_prefix, kwargs_color,

            tiles_map_by_epsg, bbox_by_epsg):

        print('    Step 1: loading datasets from s3 bucket:')
        product_count = 1
        frequency_epsg_dict = {'A': {},
                               'B': {}}

        if flag_s3_bucket:
            creds = plant_isce3.load_aws_credentials('saml-pub')

            resource = boto3.resource('s3', **creds)

            my_bucket = resource.Bucket(bucket_name)

            files_iterator = my_bucket.objects.filter(Prefix=s3_prefix)
        else:
            file_list = glob.glob(self.input_file, recursive=True)
            files_iterator = [os.path.split(f) for f in file_list]

        for i, objects in enumerate(files_iterator):
            path, f = os.path.split(objects.key)

            if (not f.endswith('.h5') and not f.endswith('.png') and
                    not f.endswith('.kml')) or 'STATS' in f:
                continue

            if (self.filename_must_include is not None and
                    len(self.filename_must_include) > 0):

                for s in self.filename_must_include:

                    if s in path or s in f:

                        break
                else:

                    continue

            if (self.filename_must_not_include is not None and
                    len(self.filename_must_not_include) > 0):
                flag_found = False
                for s in self.filename_must_not_include:
                    if s in path or s in f:
                        flag_found = True
                        break
                if flag_found:
                    continue

            if (self.max_number_products is not None and
                    product_count > self.max_number_products):
                continue

            downloaded_file = os.path.join(self.step_1_directory, f)

            original_basename = os.path.splitext(f)[0]
            print('***    f:', f)
            print('***    path:', path)
            basename = os.path.splitext(f)[0]
            if basename == 'BROWSE':
                path_splitted = path.split('/')
                if path_splitted[-1] != 'qa':
                    basename = path_splitted[-1]
                else:
                    basename = path_splitted[-2]
            print('***    original_basename:', original_basename)
            print('***    basename:', basename)
            png_file = os.path.join(self.step_1_directory,
                                    f'{basename}_BROWSE.png')

            if (self.step_1_download_browse and f.endswith('.png') and
                    not os.path.isfile(png_file)):

                os.makedirs(os.path.dirname(downloaded_file),
                            exist_ok=True)
                print('    Step 1: Downloading browse file (PNG):', f)
                try:
                    my_bucket.download_file(objects.key, f)
                except BaseException:
                    print('        there was an error downloading file:', f)
                    continue
                os.rename(f, png_file)
                continue
            elif f.endswith('.png'):
                print('        Browse image already downloaded')
                continue

            kml_file = os.path.join(self.step_1_directory,
                                    f'{basename}_BROWSE.kml')

            if (self.step_1_download_kml and f.endswith('.kml') and
                    not os.path.isfile(kml_file)):

                os.makedirs(os.path.dirname(downloaded_file),
                            exist_ok=True)
                print('    Step 1: Downloading browse file:', f)
                try:
                    my_bucket.download_file(objects.key, f)

                except BaseException:
                    print('        there was an error downloading file:', f)
                    continue
                substitute_in_file(
                    f, kml_file, [f'{original_basename}.png',
                                  'overlay image'],
                    [f'{basename}_BROWSE.png', basename])
                os.remove(f)
                continue
            elif f.endswith('.kml'):
                print('        KML image already downloaded')
                continue

            print(f'## {i} - Product {product_count}: {basename}')

            s3_product_path = os.path.join('s3://', bucket_name, path, f)
            vsis3_product_path = s3_product_path.replace('s3://', '/vsis3/')
            if flag_s3_bucket:

                kwargs = {
                    'secret_id': np.bytes_(creds["aws_access_key_id"]),
                    'secret_key': np.bytes_(creds["aws_secret_access_key"]),
                    'aws_region': np.bytes_(creds["region_name"])
                }

                if creds["aws_session_token"]:
                    kwargs["session_token"] = \
                        np.bytes_(creds["aws_session_token"])

                h5_obj = h5py.File(s3_product_path, driver='ros3',
                                   **kwargs)

            else:
                h5_obj = h5py.File(os.join(path, f), swmr=True)

            current_file_product_type = get_product_type(h5_obj)
            current_product_level = get_product_level(h5_obj)
            if self.product_type is not None:
                if self.product_type != current_file_product_type:
                    continue

            kwargs_product_data_to_backscatter = {}
            if current_file_product_type == 'GSLC':
                kwargs_product_data_to_backscatter['square'] = True

            list_of_frequencies_orig = \
                h5_obj['/science/LSAR/identification/listOfFrequencies']
            bounding_polygon = h5_obj['/science/LSAR/identification/'
                                      'boundingPolygon'][()].decode()

            if self.bbox:

                bbox_min_lat, bbox_max_lat, bbox_min_lon, bbox_max_lon = \
                    self.bbox

                self.print('selection bbox:')
                with plant.PlantIndent():
                    self.print(f'bbox_min_lat: {bbox_min_lat}')
                    self.print(f'bbox_min_lon: {bbox_min_lon}')
                    self.print(f'bbox_max_lat: {bbox_max_lat}')
                    self.print(f'bbox_max_lon: {bbox_max_lon}')

                outer_ring = ogr.Geometry(ogr.wkbLinearRing)
                outer_ring.AddPoint(bbox_max_lon, bbox_min_lat)
                outer_ring.AddPoint(bbox_max_lon, bbox_max_lat)
                outer_ring.AddPoint(bbox_min_lon, bbox_max_lat)
                outer_ring.AddPoint(bbox_min_lon, bbox_min_lat)
                outer_ring.CloseRings()
                outer_polygon_ogr = ogr.Geometry(ogr.wkbPolygon)
                outer_polygon_ogr.AddGeometry(outer_ring)

                product_polygon = ogr.CreateGeometryFromWkt(bounding_polygon)
                min_lon, max_lon, min_lat, max_lat = \
                    product_polygon.GetEnvelope()

                self.print('product extents:')
                with plant.PlantIndent():
                    self.print(f'min_lat: {min_lat}')
                    self.print(f'min_lon: {min_lon}')
                    self.print(f'max_lat: {max_lat}')
                    self.print(f'max_lon: {max_lon}')

                if not outer_polygon_ogr.Contains(product_polygon):
                    print('Product is outside bbox')
                    continue

            list_of_frequencies = [freq.decode()
                                   for freq in list_of_frequencies_orig]
            list_of_frequencies_dict = {}

            for frequency in list_of_frequencies:

                if current_product_level == 'L1':
                    h5_path = (f'/science/LSAR/{current_file_product_type}/'
                               f'swaths/frequency{frequency}/'
                               'listOfPolarizations')
                else:
                    h5_path = (f'/science/LSAR/{current_file_product_type}/'
                               f'grids/frequency{frequency}/'
                               'listOfPolarizations')

                if h5_path not in h5_obj:
                    continue

                pol_list_orig = h5_obj[h5_path]

                pol_list = [pol.decode() for pol in pol_list_orig]
                list_of_frequencies_dict[frequency] = pol_list

            if current_product_level == 'L2':
                epsg = str(get_product_epsg(h5_obj, current_file_product_type))
            else:
                epsg = str(4326)

            h5_obj.close()
            del h5_obj

            update_tiles_map_dict(tiles_map_by_epsg, bbox_by_epsg,
                                  bounding_polygon, epsg)

            output_dir = self.step_2_directory

            os.makedirs(output_dir, exist_ok=True)

            if (self.step_1_download_hdf5 and
                    not os.path.isfile(downloaded_file)):

                os.makedirs(os.path.dirname(downloaded_file),
                            exist_ok=True)
                print('    Step 1: Downloading file:', f)
                try:
                    my_bucket.download_file(objects.key, f)
                    os.rename(f, downloaded_file)
                    product_count += 1
                except BaseException:
                    print('        there was an error downloading file:', f)
                    continue
            elif os.path.isfile(downloaded_file):
                print('        HDF5 already downloaded:', f)
                product_count += 1
            else:
                print('        HDF5 not found/downloaded:', f)
                downloaded_file = None

            for frequency, pols in list_of_frequencies_dict.items():

                if frequency not in frequency_epsg_dict.keys():
                    continue

                if downloaded_file is None:
                    downloaded_file = vsis3_product_path
                    print('        using remote reference (vsis3):',
                          downloaded_file)

                self.run_process_native_coordinates_freq(
                    kwargs_color, kwargs_product_data_to_backscatter,
                    frequency_epsg_dict, downloaded_file, basename, epsg,
                    output_dir, frequency, pols, current_product_level)

        return frequency_epsg_dict

    def run_process_native_coordinates_freq(
            self, kwargs_color, kwargs_product_data_to_backscatter,
            frequency_epsg_dict, downloaded_file, basename, epsg, output_dir,
            frequency, pols, current_product_level):

        nlooks_y, nlooks_x = self.get_nlooks(frequency=frequency)

        if nlooks_y != 1 or nlooks_x != 1:
            suffix = (f'_{frequency}_ml_{nlooks_y}_{nlooks_x}')
        else:
            suffix = f'_{frequency}'

        output_file = os.path.join(output_dir, basename + suffix + '.tif')
        output_kmz = os.path.join(output_dir, basename + suffix + '.kmz')
        output_png = os.path.join(output_dir, basename + suffix + '.png')

        if (self.step_2_generate_cog_rgb and not os.path.isfile(output_file)):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            filter_method(
                input_ref,

                nlooks_x=nlooks_x,
                nlooks_y=nlooks_y,
                output_file=output_file,
                force=True,

                **kwargs_product_data_to_backscatter)

        for band, pol in enumerate(pols):
            output_file_pol = \
                os.path.join(output_dir, f'{basename}{suffix}_{pol}.tif')
            if (self.step_2_generate_cog and
                    not os.path.isfile(output_file_pol)):
                input_ref = f'NISAR:{downloaded_file}:{frequency}'
                filter_method(
                    input_ref,

                    nlooks_x=nlooks_x,
                    nlooks_y=nlooks_y,
                    output_file=output_file_pol,
                    force=True,
                    band=band,

                    **kwargs_product_data_to_backscatter)

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

        if (self.step_2_generate_kmz and not os.path.isfile(output_kmz) and
                os.path.isfile(output_file) and current_product_level == 'L2'):
            self.util(output_file, output_file=output_kmz, force=True)

        elif (self.step_2_generate_kmz and not os.path.isfile(output_kmz) and
              current_product_level == 'L2'):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            filter_method(input_ref,

                          nlooks_x=nlooks_x,
                          nlooks_y=nlooks_y,
                          output_file=output_kmz, force=True,

                          **kwargs_product_data_to_backscatter,
                          **kwargs_color)
        elif self.step_2_generate_kmz and not os.path.isfile(output_kmz):
            if os.path.isfile(output_file):
                rslc_file = output_file
            else:
                rslc_file = downloaded_file

            plant_isce3.geocode(rslc_file,

                                dem_file=self.dem_file,
                                output_file=output_kmz, force=True,

                                **kwargs_color)

        if (self.step_2_generate_png and not os.path.isfile(output_png) and
                os.path.isfile(output_file)):
            self.util(output_file, output_file=output_png, force=True)
        elif (self.step_2_generate_png and not os.path.isfile(output_png)):
            input_ref = f'NISAR:{downloaded_file}:{frequency}'
            filter_method(input_ref,

                          nlooks_x=nlooks_x,
                          nlooks_y=nlooks_y,
                          cmap_max=self.cmap_max,
                          cmap_min=self.cmap_min,
                          output_file=output_png, force=True,

                          **kwargs_product_data_to_backscatter,
                          **kwargs_color)

    def create_tiles(self, flag_generate_tiles, flag_generate_tiles_kmz,
                     flag_generate_tiles_rgb_kmz,
                     flag_generate_tiles_ab_kmz, flag_vrts,
                     frequency, orbit_pass_direction_str,
                     min_lat, max_lat, min_lon, max_lon,
                     tiles_map, flag_last_pol, suffix_list, suffix_rgb,

                     list_of_output_files,

                     output_dir_prefix,
                     suffix=''):

        min_lat = int(min_lat)
        max_lat = int(np.ceil(max_lat))
        min_lon = int(min_lon)
        max_lon = int(np.ceil(max_lon))

        print('Extents:')
        print('    min lon:', min_lon)
        print('    max lon:', max_lon)
        print('    min lat:', min_lat)
        print('    max lat:', max_lat)

        file_list = []

        for lat in range(min_lat, max_lat + 1):
            sn_str = 'S' if lat < 0 else 'N'
            lat_str = f'{sn_str}{abs(lat):02d}_00'

            for lon in range(min_lon, max_lon + 1):
                we_str = 'W' if lon < 0 else 'E'
                lon_str = f'{we_str}{abs(lon):03d}_00'

                tile_file = os.path.join(
                    f'{output_dir_prefix}',
                    f'{self.product_type}_{lat_str}_{lon_str}{suffix}.tif')

                tile_kmz_file = os.path.join(
                    f'{output_dir_prefix}_tiles_kmz',
                    f'{self.product_type}_{lat_str}_{lon_str}{suffix}.kmz')

                tile_rgb_kmz_file = os.path.join(
                    f'{output_dir_prefix}_tiles_kmz',
                    f'{self.product_type}_{lat_str}_{lon_str}{suffix_rgb}.kmz')

                tile_kmz_ab_hh_file = os.path.join(
                    f'{output_dir_prefix}_tiles_ab',
                    f'{self.product_type}_{lat_str}_{lon_str}_AB'
                    f'_HH{orbit_pass_direction_str}.kmz')

                if flag_generate_tiles and not os.path.isfile(tile_file):
                    try:
                        plant.mosaic(*list_of_output_files,
                                     output_file=tile_file,
                                     bbox=[lat, lat + 1, lon, lon + 1],
                                     step=res_deg_dict[frequency],
                                     force=True, in_null='nan',
                                     out_null='nan', of=cog_str,
                                     interp='average',
                                     out_projection='wgs84')

                        file_list.append(tile_file)
                    except BaseException:

                        pass

                elif os.path.isfile(tile_file):
                    file_list.append(tile_file)

                if not os.path.isfile(tile_file):
                    continue

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

        vrt_file = f'{output_dir_prefix}/mosaic{suffix}.vrt'
        if flag_vrts:
            os.makedirs(output_dir_prefix, exist_ok=True)

            if os.path.isfile(vrt_file):
                os.remove(vrt_file)
            gdal.BuildVRT(vrt_file, file_list, srcNodata='nan',
                          VRTNodata='nan',
                          outputBounds=[min_lon, min_lat, max_lon, max_lat])

            print('        file saved:', vrt_file)
            add_overviews_vrt(vrt_file)
            print(f'        file updated: {vrt_file} (added overviews)')

        return vrt_file


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
        print(f'    replacing "{old_substring}" with "{new_substring}"')
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


def get_product_type(h5_obj):

    product_type = \
        h5_obj['/science/LSAR/identification/productType'][()]
    if not isinstance(product_type, str):
        product_type = product_type.decode()

    return product_type


def get_product_level(h5_obj):

    product_level = \
        h5_obj['/science/LSAR/identification/productLevel'][()]
    if not isinstance(product_level, str):
        product_level = product_level.decode()

    return product_level


def get_product_epsg(h5_obj, product_type):

    list_of_frequencies = h5_obj['/science/LSAR/identification/'
                                 'listOfFrequencies']
    first_frequency = list_of_frequencies[0].decode()
    projection = h5_obj[f'/science/LSAR/{product_type}/grids/'
                        f'frequency{first_frequency}/projection']
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


def update_tiles_map_dict(tiles_map_by_epsg,
                          bbox_by_epsg, bounding_polygon, epsg):
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

    polygon_geometry = ogr.CreateGeometryFromWkt(bounding_polygon)
    (min_lon, max_lon, min_lat, max_lat) = polygon_geometry.GetEnvelope()
    print('    min_lon, max_lon, min_lat, max_lat:',
          min_lon, max_lon, min_lat, max_lat)

    epsg_min_lon = min([epsg_min_lon, min_lon])
    epsg_max_lon = max([epsg_max_lon, max_lon])
    epsg_min_lat = min([epsg_min_lat, min_lat])
    epsg_max_lat = max([epsg_max_lat, max_lat])

    lat_index_beg = 180 - (int(np.ceil(max_lat)) + 90)
    lat_index_end = 180 - (int(np.ceil(min_lat)) + 90) + 1
    lon_index_beg = int(min_lon) + 180
    lon_index_end = int(max_lon) + 180 + 1
    epsg_tiles_map[lat_index_beg:lat_index_end,
                   lon_index_beg:lon_index_end] = 1

    tiles_map_by_epsg[epsg] = epsg_tiles_map
    bbox_by_epsg[epsg] = epsg_min_lon, epsg_max_lon, epsg_min_lat, epsg_max_lat

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

    bbox_by_epsg['mosaic'] = \
        mosaic_min_lon, mosaic_max_lon, mosaic_min_lat, mosaic_max_lat


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3BatchProcessing(parser, argv)
        ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
