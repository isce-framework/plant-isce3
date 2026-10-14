#!/usr/bin/env python3
import gc

import shutil
from osgeo import gdal
import numpy as np

import os

from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation

import json
import os

import fiona
import geopandas as gpd
import pandas as pd
from shapely import wkt

import plant
import plant_isce3
from plant_isce3.readers import open_product

PSP_NULL = 0

image_thresholds_db = {
    'HH': [plant.get_inv_db(-15),
           plant.get_inv_db(-4)],

    'HV': [plant.get_inv_db(-20),
           plant.get_inv_db(-9)],

    'VH': [plant.get_inv_db(-20),
           plant.get_inv_db(9)],

    'VV': [plant.get_inv_db(-16),
           plant.get_inv_db(-5)]
}


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            dem_file=1,
                            input_file=1,
                            default_options=1,
                            multilook=1,

                            output_file=1)

    plant_isce3.add_arguments(parser,
                              frequency=1,
                              nlooks_by_frequency=1)

    parser.add_argument('--abs-cal-factor',
                        dest='abs_cal_factor',

                        type=str,
                        help='Absolute radiometric calibration factor')

    parser.add_argument('--minimum-azimuth-valid-percentage',
                        dest='minimum_azimuth_valid_percentage',
                        default=25,
                        type=float,
                        help=('Minimum percentage of valid pixels in the'
                              ' azimuth direction required for a pixel to'
                              ' be considered valid.'))

    parser.add_argument('--pol-list',
                        dest='pol_list',
                        nargs='+',
                        type=str,
                        help='Polarization list')

    parser.add_argument('--rg-filter-size',
                        dest='rg_filter_size',
                        default=51,
                        type=int,
                        help='Filter size')

    parser.add_argument('--divisor-min-value-db',
                        dest='divisor_min_value_db',
                        default=-30,
                        type=float,
                        help='Divisor min value in dB')

    parser.add_argument('--min-backscatter-value',
                        dest='min_backscatter_value',
                        default=0.00001,

                        type=float,
                        help='Minimum backscatter value')

    parser.add_argument('--water-mask-a',
                        dest='water_mask_a',
                        type=str,
                        help='Water mask from WorldCover (freq. A)')

    parser.add_argument('--water-mask-b',
                        dest='water_mask_b',
                        type=str,
                        help='Water mask from WorldCover (freq. B)')

    parser.add_argument('--min-range-index',
                        dest='min_range_index',

                        type=int,
                        help='Minimum range index')

    parser.add_argument('--max-range-index',
                        dest='max_range_index',

                        type=int,
                        help='Maximum range index')

    parser.add_argument('--profile-filter-size',
                        dest='profile_filter_size',
                        default=None,
                        type=int,
                        help='Profile filter size. "-1" to disable filtering.')

    parser.add_argument('--ignore-noise',
                        action='store_true',
                        dest='flag_ignore_noise',
                        help='Ignore noise in profile calculation')

    parser.add_argument('--add-error-fill',
                        action='store_true',
                        dest='flag_add_error_fill',
                        help='Flag add error fill')

    parser.add_argument('--generate-elevation-profiles',
                        action='store_true',
                        dest='flag_generate_elevation_profiles',
                        help='Generate elevation profiles')

    parser.add_argument('--radiometric-terrain-convention-a',
                        dest='radiometric_terrain_convention_a',
                        default='gamma0',
                        type=str,
                        choices=['gamma0', 'beta0', 'gamma0-psi'],
                        help='Radiometric terrain correction convention'
                        ' for frequency A'
                        ' (options: gamma0, beta0, gamma0-psi)')

    parser.add_argument('--radiometric-terrain-convention-b',
                        dest='radiometric_terrain_convention_b',
                        default='gamma0',
                        type=str,
                        choices=['gamma0', 'beta0', 'gamma0-psi'],
                        help='Radiometric terrain correction convention'
                        ' for frequency B'
                        ' (options: gamma0, beta0, gamma0-psi)')

    parser.add_argument('--worldcover',
                        '--worldcover-file',
                        dest='worldcover',
                        type=str,
                        help='WorldCover land cover map for selecting'
                        ' vegetated areas')

    parser.add_argument('--only-eap-stats',
                        action='store_true',
                        dest='flag_only_eap_stats',
                        help=('Process EAP statistics only (skip EAP analysis'
                              ' and plot generation)'))

    parser.add_argument('--skip-existent-eap-stats',
                        action='store_true',
                        dest='flag_skip_existent_eap_stats',
                        help='Skip calculating existent EAP stats')

    parser.add_argument('--minimum-vegetation-fraction',
                        dest='minimum_vegetation_fraction',

                        type=float,
                        help='Minimum vegetation fraction.')

    parser.add_argument('--out-eap-stats-file',
                        '--output-eap-stats-file',
                        dest='output_eap_stats_file',
                        type=str,
                        help='Output GeoPandas EAP stats file')

    parser.add_argument('--topo-dir-a',
                        '--topo-file-a',
                        dest='topo_dir_a',
                        type=str,
                        help='File or directory containing'
                        ' geolocation/topographic files for frequency A.')

    parser.add_argument('--topo-dir-b',
                        '--topo-file-b',
                        dest='topo_dir_b',
                        type=str,
                        help='File or directory containing'
                        ' geolocation/topographic files for frequency B.')

    parser.add_argument('--profiles-directory',
                        type=str,
                        dest='profiles_directory',

                        help='Input/output profiles directory')

    parser.add_argument('--temporary-images-directory',
                        '--temp-images-directory',
                        type=str,
                        dest='images_directory',

                        help='Temporary images directory')

    parser.add_argument('--read-existing-profiles',
                        '--load-existing-profiles',
                        action='store_true',
                        dest='flag_read_existing_profiles',
                        help='Read existing profiles from profiles directory')

    parser.add_argument('--read-processed-backscatter-image',
                        '--load-processed-backscatter-image',
                        action='store_true',
                        dest='flag_load_processed_backscatter_image',
                        help='Load processed backscatter image')

    parser.add_argument('--save-multilooked-backscatter-image',
                        action='store_true',
                        dest='flag_save_multilooked_backscatter_image',
                        help='Save multilooked backscatter image')

    parser.add_argument('--save-processed-backscatter-image',
                        action='store_true',
                        dest='flag_save_processed_backscatter_image',
                        help='Save processed backscatter image')

    parser.add_argument('--save-multilooked-backscatter-png',
                        action='store_true',
                        dest='flag_save_multilooked_backscatter_png',
                        help='Save multilooked backscatter image as PNG')

    parser.add_argument('--save-processed-backscatter-png',
                        action='store_true',
                        dest='flag_save_processed_backscatter_png',
                        help='Save processed backscatter image as PNG')

    parser.add_argument('--save-profile-plot-png',
                        action='store_true',
                        dest='flag_save_plot',
                        help='Save slant-range profile plot')

    parser.add_argument('--create-plots-with-predefined-thresholds',
                        action='store_true',
                        dest='flag_create_plots_with_predefined_thresholds',
                        help=('Create plots with predefined thresholds for'
                              ' better visual comparison between profiles'))

    parser.add_argument('--png-prefix',
                        type=str,
                        dest='prefix',
                        default='',
                        help='Prefix for output files')

    parser.add_argument('--profile-max-in-db',
                        type=float,
                        dest='profile_max_in_db',
                        help='Maximum value for profile in dB')

    parser.add_argument('--profile-min-in-db',
                        type=float,
                        dest='profile_min_in_db',
                        help='Minimum value for profile in dB')

    return parser


def overwrite_dataset_check(element_name, force=None, element_str='file'):

    if plant.plant_config.flag_all or force:
        return True
    if plant.plant_config.flag_never:
        return False
    while 1:
        res = plant.get_keys(f'The {element_str} {element_name} already'
                             ' exists. Would you like to overwrite'
                             ' it? ([y]es/[n]o)/[A]ll/[N]one ')
        if res == 'n':
            return False
        elif res == 'N':
            plant.plant_config.flag_never = True
            return False
        elif res == 'y':
            return True
        elif res == 'A':
            plant.plant_config.flag_all = True
            return True


def smooth_edges(x, transition_width, offset_left=0, offset_right=0,
                 flag_left=True, flag_right=True):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if transition_width <= 0:
        return x.copy()
    if 2 * transition_width > n:
        raise ValueError("transition_width is too large for vector length")

    y = x.copy()

    if offset_left + transition_width > n:
        raise ValueError("offset_left + transition_width is too large"
                         " for vector length")

    t = np.linspace(0, 1, transition_width)
    ramp = 0.5 * (1 - np.cos(np.pi * t))

    if flag_left:
        print('***   transition width:', transition_width)
        print('***   offset left:', offset_left)
        end_value = x[offset_left + transition_width - 1]
        print('***   left end_value:', end_value)

        y[0: offset_left] = 1

        y[offset_left:offset_left + transition_width] = \
            (1 - ramp) * 1 + ramp * end_value

    if flag_right:
        print('***   transition width:', transition_width)
        print('***   offset right:', offset_right)
        y[offset_right:] = 1
        start_value = x[offset_right - transition_width]
        print('***   right start_value:', start_value)

        y[-transition_width + offset_right:offset_right] = \
            (1 - ramp[::-1]) * start_value + ramp[::-1] * 1
    return y


def update_vegetation_gpkg(
    cycle: int,
    track: int,
    frame: int,
    frequency: str,
    max_rfi_likelihood: float,
    vegetation_fraction: float,
    acquisition_mode_list: list,
    wkt_polygon: str,
    filename: str,
    layer: str = "vegetation",
    hh: float = None,
    hv: float = None,
    vh: float = None,
    vv: float = None,
) -> gpd.GeoDataFrame:

    print('executing: update_vegetation_gpkg()')

    geometry = wkt.loads(wkt_polygon)

    if geometry.geom_type != "Polygon":
        raise ValueError(
            f"Expected a Polygon, got {geometry.geom_type}."
        )

    record = {
        "cycle": cycle,
        "track": track,
        "frame": frame,
        "frequency": frequency,
        "max_rfi_likelihood": max_rfi_likelihood,
        "vegetation_fraction": vegetation_fraction,
        "acquisition_mode_list": json.dumps(acquisition_mode_list),
        "HH": hh,
        "HV": hv,
        "VH": vh,
        "VV": vv,
        "geometry": geometry,
    }

    new_gdf = gpd.GeoDataFrame(
        [record],
        geometry="geometry",
        crs="EPSG:4326",
    )

    key_columns = ["cycle", "track", "frame", "frequency"]

    file_exists = os.path.exists(filename)
    layer_exists = (
        file_exists and layer in fiona.listlayers(filename)
    )

    if layer_exists:
        gdf = gpd.read_file(filename, layer=layer)

        if gdf.crs is None:
            raise ValueError(
                f"The existing layer '{layer}' has no CRS."
            )

        gdf = gdf.to_crs("EPSG:4326")

        mask = pd.Series(True, index=gdf.index)

        for column in key_columns:
            mask &= gdf[column] == record[column]

        gdf = gdf.loc[~mask].copy()

        gdf = gpd.GeoDataFrame(
            pd.concat([gdf, new_gdf], ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        gdf = new_gdf

    if layer_exists:
        fiona.remove(filename, driver="GPKG", layer=layer)

    gdf.to_file(
        filename,
        layer=layer,
        driver="GPKG",
        mode="a" if file_exists else "w",
        index=False,
    )

    return gdf


class PlantIsce3RslcEapAnalysis(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        if self.output_file:
            ret = self.overwrite_file_check(self.output_file)
            if not ret:
                self.print('Operation cancelled.', 1)
                return

            if self.input_file != self.output_file:

                input_file_obj = plant.h5py_file_wrapper(self.input_file, 'r')
                input_file_obj.close()
                shutil.copyfile(self.input_file, self.output_file)

            reference_file = self.output_file
            reference_file_opening_mode = 'a'
        else:
            reference_file = self.input_file
            reference_file_opening_mode = 'r'

        nisar_product_obj = open_product(reference_file)

        assert nisar_product_obj.productType == 'RSLC'

        product_path = nisar_product_obj.ProductPath

        plant_product_obj = self.load_product()

        flag_has_input_data_exception = \
            plant_product_obj.get_nisar_identification_scalar(
                'hasInputDataException')

        data_exception_str = \
            f'data exception: {flag_has_input_data_exception}'

        if self.frequency is None:
            freq_pol_dict = nisar_product_obj.polarizations
        else:
            freq_pol_dict = {self.frequency:
                             nisar_product_obj.polarizations[self.frequency]}

        if self.percentile is None and self.worldcover is not None:
            print('worldcover provided but percentile not set,'
                  ' defaulting to 50th percentile')
            self.percentile = 50
        elif self.percentile is None:
            print('percentile not set, defaulting to 90th percentile')
            self.percentile = 90

        print('## frequencies to process:', ', '.join(
            list(freq_pol_dict.keys())))

        with plant.h5py_file_wrapper(
            reference_file, reference_file_opening_mode, swmr=True) as \
                rslc_obj:

            for freq in freq_pol_dict.keys():

                print('## processing frequency:', freq)

                wkt_polygon = nisar_product_obj.identification.boundingPolygon

                dict_filename = self.parse_nisar_product_filename(
                    self.input_file)

                cycle = None
                track = None
                frame = None
                vegetation_fraction = None

                if 'cycle_number' in dict_filename.keys():
                    cycle = dict_filename['cycle_number']
                if 'track_number' in dict_filename.keys():
                    track = dict_filename['track_number']

                if 'frame_number' in dict_filename.keys():
                    frame = dict_filename['frame_number']

                if self.worldcover:

                    polygon = wkt.loads(wkt_polygon)

                    mean_latitude = np.mean([coord[1] for coord in
                                             polygon.exterior.coords])

                    print(f"Mean latitude: {mean_latitude}")
                    if mean_latitude <= -60 or mean_latitude >= 84:
                        print('ERROR worldcover does not cover the give area')
                        return

                if (self.flag_only_eap_stats and self.flag_skip_existent_eap_stats and
                        os.path.isfile(self.output_eap_stats_file)):
                    layer = 'vegetation'
                    gdf = gpd.read_file(
                        self.output_eap_stats_file, layer=layer)

                    mask = (
                        (gdf["cycle"] == cycle)
                        & (gdf["track"] == track)
                        & (gdf["frame"] == frame)
                        & (gdf["frequency"] == freq)
                    )

                    if mask.any():

                        print(
                            '## Found a stats record matching the current EAP'
                            ' analysis. Skipping.')
                        continue

                flag_flat_pol = (
                    not self.flag_read_existing_profiles or
                    self.flag_generate_elevation_profiles or
                    self.flag_save_multilooked_backscatter_image or
                    self.flag_save_processed_backscatter_image or
                    self.flag_save_multilooked_backscatter_png or
                    self.flag_save_multilooked_backscatter_png or
                    self.flag_save_plot)

                if self.abs_cal_factor is not None:
                    abs_cal_factor = \
                        list(plant.read_image(self.abs_cal_factor).image[0])
                else:
                    abs_cal_factor = None

                nlooks_y, nlooks_x = self.get_nlooks(frequency=freq)

                if nlooks_y > 1 or nlooks_x > 1:
                    print('    nlooks_y:', nlooks_y)
                    print('    nlooks_x:', nlooks_x)
                    suffix_ml = f'_ml_{nlooks_y}_{nlooks_x}'
                else:
                    suffix_ml = ''

                if self.profiles_directory is not None:
                    profiles_directory = self.profiles_directory
                else:
                    profiles_directory = \
                        ('rslc_eap_analysis_freq_' + freq + suffix_ml)

                if self.images_directory is not None:
                    images_directory = self.images_directory
                else:
                    images_directory = os.path.join(self.profiles_directory,
                                                    'images')

                if (freq == 'A' and
                        'gamma' in self.radiometric_terrain_convention_a):
                    radiometric_terrain_convention = 'gamma0'
                elif (freq == 'B' and
                        'gamma' in self.radiometric_terrain_convention_b):
                    radiometric_terrain_convention = 'gamma0'
                else:
                    radiometric_terrain_convention = 'beta0'

                gamma_to_beta_factor = None
                mask_non_forest = None

                if self.pol_list is not None:
                    pol_list = self.pol_list
                else:
                    pol_list = freq_pol_dict[freq]

                if (flag_flat_pol and not self.flag_only_eap_stats and
                    ((freq == 'A' and
                      self.radiometric_terrain_convention_a == 'gamma0') or
                     (freq == 'B' and
                      self.radiometric_terrain_convention_b == 'gamma0'))):
                    print('computing gamma0 using ISCE3 RTC module for'
                          f' frequency {freq}')

                    output_rtc_anf_file = os.path.join(
                        images_directory,
                        f'rtc_anf_{freq}{suffix_ml}.tif')

                    if (self.output_skip_if_existent and
                            os.path.isfile(output_rtc_anf_file)):
                        print(f'ISCE3 RTC output file {output_rtc_anf_file}'
                              ' already exists, skipping RTC computation for'
                              f' frequency {freq}')

                    else:
                        plant_isce3.rtc(
                            self.input_file,
                            dem=self.dem_file,
                            simulate=True,
                            frequency=freq,
                            rtc_min_value_db=-1,
                            output_file=output_rtc_anf_file,

                            nlooks_x=nlooks_x,
                            nlooks_y=nlooks_y,
                            force=self.force)

                    gamma_to_beta_factor = \
                        plant.read_image(output_rtc_anf_file).image

                if (flag_flat_pol and self.worldcover is not None or
                    (freq == 'A' and self.radiometric_terrain_convention_a ==
                        'gamma0-psi') or
                        (freq == 'B' and
                         self.radiometric_terrain_convention_b ==
                         'gamma0-psi')):

                    if freq == 'A' and self.topo_dir_a:
                        topo_dir = self.topo_dir_a
                    elif freq == 'B' and self.topo_dir_b:
                        topo_dir = self.topo_dir_b
                    else:
                        topo_dir = os.path.join(
                            images_directory,
                            f'topo_dir_{freq}{suffix_ml}')

                    psi_file = f'{topo_dir}/localPsi.tif'
                    local_inc_file = \
                        f'{topo_dir}/localInc.tif'
                    x_file = f'{topo_dir}/x.tif'
                    y_file = f'{topo_dir}/y.tif'
                    required_files_list = []
                    flag_local_incidence_angle = False
                    flag_projection_angle = False
                    flag_x = False
                    flag_y = False
                    if self.worldcover is not None:
                        required_files_list += [x_file, y_file]
                        flag_x = True
                        flag_y = True
                    if ((freq == 'A' and
                         self.radiometric_terrain_convention_a ==
                        'gamma0-psi') or
                        (freq == 'B' and
                         self.radiometric_terrain_convention_b ==
                            'gamma0-psi')):
                        required_files_list += [psi_file, local_inc_file]
                        flag_local_incidence_angle = True
                        flag_projection_angle = True

                    flag_run_topo = (not self.output_skip_if_existent or
                                     any(not os.path.isfile(f)
                                         for f in required_files_list))

                    if flag_run_topo:
                        print('generating topo directory for frequency', freq)
                        plant_isce3.topo(
                            self.input_file,
                            output_dir=topo_dir,
                            dem_file=self.dem_file,
                            frequency=freq,
                            nlooks_x=nlooks_x,
                            nlooks_y=nlooks_y,
                            out_local_incidence_angle=flag_local_incidence_angle,
                            out_projection_angle=flag_projection_angle,
                            out_x=flag_x,
                            out_y=flag_y)

                    if self.worldcover is not None:
                        print('loading WorldCover land cover map for'
                              ' frequency', freq)
                        projected_worldcover_file = \
                            f'{topo_dir}/projected_worldcover.tif'
                        if (not self.output_skip_if_existent or
                                not os.path.isfile(projected_worldcover_file)):
                            plant.slantrange(
                                self.worldcover,
                                output_file=projected_worldcover_file,
                                topo_dir=topo_dir,
                                force=self.force)

                        mask_non_forest = plant.read_image(
                            projected_worldcover_file).image != 10

                        if self.minimum_vegetation_fraction is not None:
                            non_forest_pixels = np.count_nonzero(
                                mask_non_forest)
                            all_pixels = mask_non_forest.size

                            vegetation_fraction = (
                                (all_pixels - non_forest_pixels) / all_pixels
                            )

                            print(
                                f'Number of non-forest pixels: {non_forest_pixels}')
                            print(f'Total number of pixels: {all_pixels}')
                            print(
                                f'Vegetation fraction: {vegetation_fraction}')

                            if (vegetation_fraction <
                                    self.minimum_vegetation_fraction):
                                print(
                                    'WARNING: Vegetation fraction'
                                    f' {vegetation_fraction:.3f} '
                                    f'is below the minimum threshold '
                                    f'{self.minimum_vegetation_fraction:.3f}')

                                acquisition_mode_list = self._get_acquisition_mode_list(
                                    nisar_product_obj)

                                max_rfi_likelihood = self.get_max_rfi_likelihood(
                                    freq, pol_list)

                                if self.output_eap_stats_file:
                                    update_vegetation_gpkg(
                                        cycle,
                                        track,
                                        frame,
                                        freq,
                                        max_rfi_likelihood,
                                        vegetation_fraction,
                                        acquisition_mode_list,
                                        wkt_polygon,
                                        self.output_eap_stats_file,
                                        layer="vegetation"
                                    )
                                return

                    if ((freq == 'A' and
                         self.radiometric_terrain_convention_a ==
                         'gamma0-psi') or
                        (freq == 'B' and
                         self.radiometric_terrain_convention_b ==
                            'gamma0-psi')):

                        cos_psi = \
                            np.cos(np.radians(
                                plant.read_image(psi_file).image))
                        cos_local_inc = \
                            np.cos(np.radians(
                                plant.read_image(local_inc_file).image))
                        gamma_to_beta_factor = cos_local_inc / cos_psi
                        del cos_psi
                        del cos_local_inc

                radar_grid = plant_product_obj.get_radar_grid(frequency=freq)
                radar_grid_ml = plant_product_obj.get_radar_grid_ml(
                    frequency=freq)

                print('*** radar grid length:', radar_grid.length)
                print('*** radar grid width:', radar_grid.width)

                print('*** radar grid ml length:', radar_grid_ml.length)
                print('*** radar grid ml width:', radar_grid_ml.width)

                print('pol list:', pol_list)
                display_profiles_dict = {}
                display_elevation_profiles_dict = {}
                self.input_raster = None
                if not self.flag_only_eap_stats:
                    for pol_count, pol in enumerate(pol_list):

                        if flag_flat_pol:
                            self.flat_pol(profiles_directory,
                                          images_directory,
                                          nisar_product_obj, plant_product_obj,
                                          product_path, rslc_obj,
                                          nlooks_y, nlooks_x,
                                          freq, abs_cal_factor,
                                          gamma_to_beta_factor,
                                          mask_non_forest,
                                          pol_count, pol,
                                          radar_grid, radar_grid_ml,
                                          display_profiles_dict,
                                          display_elevation_profiles_dict,

                                          suffix_ml)

                        if not self.output_file:

                            gc.collect()
                            continue

                        self.flatten_and_save_h5_pol(profiles_directory,
                                                     images_directory,
                                                     product_path,
                                                     rslc_obj,
                                                     freq,

                                                     pol,
                                                     radar_grid)

                        gc.collect()

                    self.plot_results(
                        data_exception_str,
                        freq,
                        suffix_ml,
                        profiles_directory,
                        images_directory,
                        pol_list,
                        display_profiles_dict,
                        display_elevation_profiles_dict,
                        radiometric_terrain_convention.title())

                if self.output_eap_stats_file and not self.worldcover:
                    self.print(
                        "ERROR WorldCover required to save EAP stats file")
                    return
                elif self.output_eap_stats_file:
                    print(
                        f'## Saving EAP stats file: {
                            self.output_eap_stats_file}')

                    print(f'cycle: {cycle}')
                    print(f'track: {track}')
                    print(f'frame: {frame}')

                    if vegetation_fraction is None:
                        non_forest_pixels = np.count_nonzero(mask_non_forest)
                        all_pixels = mask_non_forest.size

                        vegetation_fraction = (
                            (all_pixels - non_forest_pixels) / all_pixels
                        )

                        print(
                            f'Number of non-forest pixels: {non_forest_pixels}')
                        print(f'Total number of pixels: {all_pixels}')
                        print(f'Vegetation fraction: {vegetation_fraction}')

                    print(f'WKT polygon: {wkt_polygon}')

                    average_backscatter = {}
                    for pol in pol_list:
                        if pol.lower() not in average_backscatter.keys():
                            continue
                        average_backscatter[pol.lower()] = \
                            display_profiles_dict[pol + '_offset']

                    print(f'average backscatter: {average_backscatter}')

                    acquisition_mode_list = self._get_acquisition_mode_list(
                        nisar_product_obj)

                    max_rfi_likelihood = self.get_max_rfi_likelihood(
                        freq, pol_list)

                    print(f'maximum RFI likelihood: {max_rfi_likelihood}')

                    update_vegetation_gpkg(
                        cycle,
                        track,
                        frame,
                        freq,
                        max_rfi_likelihood,
                        vegetation_fraction,
                        acquisition_mode_list,
                        wkt_polygon,
                        self.output_eap_stats_file,
                        layer="vegetation",
                        **average_backscatter,
                    )

                    if os.path.isfile(self.output_eap_stats_file):
                        print(f'# file saved: {self.output_eap_stats_file}')
                        plant.append_output_file(self.output_eap_stats_file)

        if self.output_file:
            print(f'# file saved: {self.output_file}')
            plant.append_output_file(self.output_file)

    def get_max_rfi_likelihood(self, freq, pol_list):
        with plant.h5py_file_wrapper(self.input_file, 'r', swmr=True) as h5_obj:
            rfi_likelihood_list = []
            for pol in pol_list:
                rfi_likelihood_path = \
                    ('/science/LSAR/RSLC/metadata/'
                     'calibrationInformation/'
                     f'frequency{freq}/{pol}/rfiLikelihood')
                print(f'rfi_likelihood_path ({pol}): {rfi_likelihood_path}')
                rfi_likelihood_dataset = h5_obj[rfi_likelihood_path]
                rfi_likelihood = float(rfi_likelihood_dataset[()])
                print(f'rfi_likelihood ({pol}): {rfi_likelihood}')
                rfi_likelihood_list.append(rfi_likelihood)
            max_rfi_likelihood = max(rfi_likelihood_list)
        return max_rfi_likelihood

    def _get_acquisition_mode_list(self, nisar_product_obj):
        with plant.h5py_file_wrapper(self.input_file, 'r', swmr=True) as h5_obj:
            acquisition_mode_list = plant_isce3.get_acquisition_mode_list(
                h5_obj, nisar_product_obj)
        return acquisition_mode_list

    def plot_results(self, data_exception_str, freq, suffix_ml,
                     profiles_directory, images_directory, pol_list,
                     display_profiles_dict, display_elevation_profiles_dict,
                     radiometric_terrain_convention):

        plot_kwargs = {
            'grid_distance_major_y': 1,
            'grid_distance_minor_y': 0.25
        }

        if self.flag_save_multilooked_backscatter_png:
            min_list = []
            max_list = []
            multilooked_backscatter_png_file = os.path.join(
                profiles_directory,
                f'{self.prefix}backscatter'
                f'_{freq}_original{suffix_ml}.png')
            pol_file_list = []
            for pol in sorted(pol_list):
                if pol == 'VH' and 'HV' in pol_list:
                    continue
                pol_file = os.path.join(
                    images_directory,
                    f'backscatter'
                    f'_{freq}_original_{pol}{suffix_ml}.tif')
                if not os.path.isfile(pol_file):
                    print(f'ERROR multilooked backscatter image file'
                          f' {pol_file} not found. Consider rerunning'
                          ' with the flag'
                          ' --save-multilooked-backscatter-image'
                          ' enabled.')
                    continue
                if self.flag_create_plots_with_predefined_thresholds:
                    min_list.append(image_thresholds_db[pol][0])
                    max_list.append(image_thresholds_db[pol][1])
                pol_file_list.append(pol_file)

            image_kwargs = {}
            if len(min_list) != 0:
                image_kwargs['min'] = ', '.join(map(str, min_list))
            if len(max_list) != 0:
                image_kwargs['max'] = ', '.join(map(str, max_list))

            title = f'Beta0 Backscatter - Freq. {freq} ({data_exception_str})'

            plant.display(*pol_file_list,
                          output_file=multilooked_backscatter_png_file,
                          force=True,
                          title=title,
                          fontsize=9,
                          no_show=True,
                          crop_plots=True,

                          **image_kwargs)

        if self.flag_save_processed_backscatter_png:
            min_list = []
            max_list = []
            min_db_list = []
            max_db_list = []
            processed_backscatter_png_file = os.path.join(
                profiles_directory,
                f'{self.prefix}backscatter'
                f'_{freq}_processed{suffix_ml}.png')
            pol_file_list = []
            pol_list_sorted = sorted(pol_list)
            for pol in pol_list_sorted:
                if pol == 'VH' and 'HV' in pol_list:
                    continue
                pol_file = os.path.join(
                    images_directory,
                    f'backscatter_{freq}_processed_{pol}{suffix_ml}'
                    '.tif')
                if not os.path.isfile(pol_file):
                    print(f'ERROR processed backscatter image file'
                          f' {pol_file} not found. Consider rerunning'
                          ' with the flag'
                          ' --save-processed-backscatter-image'
                          ' enabled.')
                    continue
                if self.flag_create_plots_with_predefined_thresholds:
                    min_list.append(image_thresholds_db[pol][0])
                    max_list.append(image_thresholds_db[pol][1])

                    min_db_list.append(
                        plant.get_db(
                            image_thresholds_db[pol][0]))
                    max_db_list.append(
                        plant.get_db(
                            image_thresholds_db[pol][1]))
                pol_file_list.append(pol_file)

            image_kwargs = {}
            if len(min_list) != 0:
                image_kwargs['min'] = ', '.join(map(str, min_list))
            if len(max_list) != 0:
                image_kwargs['max'] = ', '.join(map(str, max_list))

            title = (f'{radiometric_terrain_convention} backscatter - Freq.'
                     f' {freq} ({data_exception_str})')
            plant.display(*pol_file_list,
                          output_file=processed_backscatter_png_file,
                          force=True,
                          title=title,
                          fontsize=9,
                          no_show=True,
                          crop_plots=True,

                          **image_kwargs)

            processed_backscatter_hist_png_file = os.path.join(
                profiles_directory,
                f'{self.prefix}backscatter'
                f'_{freq}_processed{suffix_ml}_hist.png')
            pol_file_list = []
            pol_list_sorted = sorted(pol_list)
            for pol in pol_list_sorted:
                pol_file = os.path.join(
                    images_directory,
                    f'backscatter_{freq}_processed_{pol}{suffix_ml}'
                    '.tif')
                if not os.path.isfile(pol_file):
                    print(f'ERROR processed backscatter image file'
                          f' {pol_file} not found. Consider rerunning'
                          ' with the flag'
                          ' --save-processed-backscatter-image'
                          ' enabled.')
                    continue
                pol_file_list.append(pol_file)

            title = (f'{radiometric_terrain_convention} backscatter - Freq.'
                     f' {freq} ({data_exception_str})')
            plant.display(*pol_file_list,
                          output_file=processed_backscatter_hist_png_file,
                          hist=True,
                          db=True,
                          force=True,
                          name=','.join(pol_list_sorted),

                          title=title,
                          fontsize=11,
                          legend=True,
                          legend_fontsize=11,
                          crop_plots=True,

                          no_show=True)

        if self.flag_generate_elevation_profiles:
            print('saving elevation profiles plot for frequency', freq)

            display_args = [
                np.array(display_elevation_profiles_dict['elevation'])]
            name_list = ['Elevation [m]']

            for pol in sorted(pol_list):
                display_args.append(
                    display_elevation_profiles_dict[pol])

                offset = display_elevation_profiles_dict[pol + '_offset']

                name_list.append(pol)

            plot_file = os.path.join(
                profiles_directory,

                f'{self.prefix}elevation_profile'
                f'_{freq}{suffix_ml}.png')

            title = (f'{radiometric_terrain_convention} backscatter'
                     f' Elevation Profiles - Freq. {freq}'
                     f' ({data_exception_str})')

            if self.profile_max_in_db is not None:
                vmax = self.profile_max_in_db
            else:
                vmax = 2.5

            if self.profile_min_in_db is not None:
                vmin = self.profile_min_in_db
            else:
                vmin = -2.5

            ylabel = (radiometric_terrain_convention +
                      ' relative backscatter [dB]')
            plant.display(*display_args,
                          output_file=plot_file,
                          force=True,
                          marker='',
                          name=','.join(name_list),
                          title=title,
                          fontsize=11,
                          legend=True,
                          legend_fontsize=11,
                          linewidth=2,

                          xlabel='Elevation [deg]',
                          ylabel=ylabel,

                          hline="-0.25,0.25",

                          stats_linestyle='dashed',
                          stats_linewidth=1,
                          stats_linecolor='red',
                          ymax=vmax,
                          ymin=vmin,
                          first_input_as_x=True,
                          crop_plots=True,
                          no_show=True,

                          **plot_kwargs)

        if self.flag_save_plot:
            print('saving profiles plot for frequency', freq)

            display_args = [
                np.array(display_profiles_dict['slant_ranges']) / 1000]
            error_fill_args = []
            name_list = ['Slant range [km]']

            for pol in sorted(pol_list):
                display_args.append(display_profiles_dict[pol])

                offset = display_profiles_dict[pol + '_offset']
                name_list.append(f"{pol} (median: {offset:.2f} dB)")
                if self.flag_add_error_fill:
                    error_fill_args.append(
                        display_profiles_dict[pol + '_stddev'])

            if self.profile_max_in_db is not None:
                vmax = self.profile_max_in_db
            else:
                vmax = 2.5

            if self.profile_min_in_db is not None:
                vmin = self.profile_min_in_db
            else:
                vmin = -2.5

            plot_file = os.path.join(
                profiles_directory,
                f'{self.prefix}slantrange_profile_{freq}{suffix_ml}'
                '.png')

            title = (f'{radiometric_terrain_convention} Backscatter Range'
                     f' Profiles - Freq. {freq} ({data_exception_str})')

            display_profile_kwargs = {}
            if self.flag_add_error_fill:
                display_profile_kwargs['error_fill'] = \
                    plant.util(*error_fill_args)
            ylabel = (radiometric_terrain_convention +
                      ' relative backscatter [dB]')
            plant.display(*display_args,
                          output_file=plot_file,
                          force=True,
                          name=','.join(name_list),
                          title=title,
                          marker='',
                          fontsize=11,
                          legend=True,
                          legend_fontsize=11,
                          linewidth=2,

                          xlabel='Slant range [km]',
                          hline="-0.25,0.25",

                          stats_linestyle='dashed',
                          stats_linewidth=1,
                          stats_linecolor='red',
                          ylabel=ylabel,
                          ymax=vmax,
                          ymin=vmin,
                          first_input_as_x=True,
                          crop_plots=True,
                          no_show=True,

                          **display_profile_kwargs,
                          **plot_kwargs)

    def flat_pol(self, profiles_directory, images_directory,
                 nisar_product_obj,
                 plant_product_obj, product_path,
                 rslc_obj, nlooks_y, nlooks_x, freq,
                 abs_cal_factor, gamma_to_beta_factor,
                 mask_non_forest,
                 pol_count, pol,
                 radar_grid, radar_grid_ml, display_profiles_dict,
                 display_elevation_profiles_dict,

                 suffix_ml):

        slant_ranges_ml = radar_grid_ml.slant_ranges

        processed_backscatter_file = os.path.join(
            images_directory,
            f'backscatter_{freq}_processed_{pol}{suffix_ml}.tif')

        multilooked_backscatter_file = os.path.join(
            images_directory,
            f'backscatter_{freq}_original_{pol}{suffix_ml}.tif')

        flag_load_processed_backscatter_image = \
            (self.flag_load_processed_backscatter_image and
             os.path.isfile(processed_backscatter_file))

        if flag_load_processed_backscatter_image:
            print(f'Loading processed backscatter image from file:'
                  f' {processed_backscatter_file}')
            backscatter_image = \
                plant.read_image(processed_backscatter_file).image
        elif (self.output_skip_if_existent and
              os.path.isfile(multilooked_backscatter_file)):
            print(f'Loading multilooked backscatter image from file:'
                  f' {multilooked_backscatter_file}')
            backscatter_image = \
                plant.read_image(multilooked_backscatter_file).image
        else:
            if self.flag_load_processed_backscatter_image:
                print(f'WARNING processed backscatter image file'
                      f' {processed_backscatter_file}'
                      ' not found, processing from input raster')
            if self.input_raster is None:
                self.input_raster = self.get_input_raster_from_nisar_product(
                    input_raster=None,
                    plant_product_obj=plant_product_obj,
                    frequency=freq)
            backscatter_image = \
                plant.util(self.input_raster, band=pol_count).image

            if 'complex' in backscatter_image.dtype.name.lower():
                backscatter_image = np.absolute(backscatter_image) ** 2

        width = backscatter_image.shape[1]

        if self.profile_filter_size is None:
            profile_filter_size = max(width // 500, 1)
        else:
            profile_filter_size = self.profile_filter_size

        metadata_dict = self.get_metadata(freq, abs_cal_factor, pol_count, pol,
                                          nlooks_y, nlooks_x,
                                          profile_filter_size)

        if not flag_load_processed_backscatter_image:

            if (self.flag_save_multilooked_backscatter_image and
                    self.output_skip_if_existent and
                    os.path.isfile(multilooked_backscatter_file)):
                print(f'Multilooked backscatter image file'
                      f' {multilooked_backscatter_file}'
                      ' already exists, skipping saving multilooked'
                      f' backscatter image for frequency {freq} polarization'
                      f' {pol}')

            elif self.flag_save_multilooked_backscatter_image:

                plant.save_image(backscatter_image,
                                 output_file=multilooked_backscatter_file,
                                 metadata=metadata_dict,
                                 force=True)

            backscatter_image = \
                self.get_masked_nisar_data_radar_coordinates(
                    nisar_product_obj, backscatter_image, freq,
                    nlooks_y, nlooks_x)

            if gamma_to_beta_factor is not None:

                if backscatter_image.shape[0] < gamma_to_beta_factor.shape[0]:
                    backscatter_image = \
                        backscatter_image[:gamma_to_beta_factor.shape[0], :]
                elif (backscatter_image.shape[0] >
                      gamma_to_beta_factor.shape[0]):
                    gamma_to_beta_factor = \
                        gamma_to_beta_factor[:backscatter_image.shape[0], :]
                if backscatter_image.shape[1] < gamma_to_beta_factor.shape[1]:
                    backscatter_image = \
                        backscatter_image[:, :gamma_to_beta_factor.shape[1]]
                elif (backscatter_image.shape[1] >
                      gamma_to_beta_factor.shape[1]):
                    gamma_to_beta_factor = \
                        gamma_to_beta_factor[:, :backscatter_image.shape[1]]

                backscatter_image /= gamma_to_beta_factor

            if mask_non_forest is not None:
                print('applying forest mask to backscatter image for frequency'
                      f' {freq} polarization {pol}')

                backscatter_image[mask_non_forest] = np.nan

            if abs_cal_factor is not None:
                print('applying absolute radiometric calibration '
                      f' factor {abs_cal_factor[pol_count]} to pol'
                      f' {pol}')
                backscatter_image *= abs_cal_factor[pol_count]

            if (self.flag_save_processed_backscatter_image and
                    self.output_skip_if_existent and
                    os.path.isfile(processed_backscatter_file)):

                print('Processed backscatter image file'
                      f' {processed_backscatter_file}'
                      ' already exists, skipping saving processed backscatter'
                      f' image for frequency {freq} polarization {pol}')

            elif self.flag_save_processed_backscatter_image:

                plant.save_image(backscatter_image,
                                 output_file=processed_backscatter_file,
                                 metadata=metadata_dict,
                                 force=True)

        if backscatter_image.shape[1] < len(slant_ranges_ml):
            slant_ranges_ml = slant_ranges_ml[:backscatter_image.shape[1]]

        if not self.flag_read_existing_profiles:
            self.get_correction_profiles(profiles_directory,
                                         nisar_product_obj,
                                         freq, abs_cal_factor,
                                         pol_count, pol,
                                         backscatter_image,
                                         radar_grid_ml,
                                         profile_filter_size,
                                         metadata_dict,
                                         display_profiles_dict)

            if self.flag_generate_elevation_profiles:
                elevation_profile, backscatter_elevation_profile, \
                    backscatter_elevation_profile_offset = \
                    self.generate_elevation_profiles(
                        profiles_directory,
                        nisar_product_obj, freq,
                        pol, backscatter_image,
                        radar_grid_ml,
                        metadata_dict,
                        profile_filter_size,

                        suffix_ml,
                        prefix='backscatter_')

                if 'elevation' not in display_elevation_profiles_dict.keys():
                    display_elevation_profiles_dict['elevation'] = \
                        elevation_profile

                display_elevation_profiles_dict[pol] = \
                    plant.get_db(backscatter_elevation_profile)

                display_elevation_profiles_dict[pol + '_offset'] = \
                    plant.get_db(backscatter_elevation_profile_offset)

    def flatten_and_save_h5_pol(self, profiles_directory, images_directory,
                                product_path, rslc_obj, freq,

                                pol, radar_grid):

        slant_ranges = np.array(radar_grid.slant_ranges)

        profile_x_min, divisor, profile_x_stddev = self.read_existing_profiles(
            profiles_directory, freq, pol, slant_ranges)

        pol_path = f'{product_path}/swaths/frequency{freq}/{pol}'

        rslc = rslc_obj[pol_path][()]

        backscatter_image = np.absolute(rslc) ** 2

        n_lines = rslc.shape[0]
        with plant.PrintProgress(n_lines) as progress_obj:
            for i in range(0, n_lines):
                progress_obj.print_progress(i)

                backscatter_image_scaled = (
                    (backscatter_image[i, :] - profile_x_min) /
                    divisor)

                if self.min_backscatter_value is not None:
                    backscatter_image_scaled = np.clip(
                        backscatter_image_scaled,
                        self.min_backscatter_value,
                        None)
                    backscatter_image_scaled[np.logical_not(
                        np.isfinite(backscatter_image_scaled))] = \
                        self.min_backscatter_value

                rslc[i, :] = (rslc[i, :] *
                              np.sqrt(backscatter_image_scaled) /
                              np.absolute(rslc[i, :]))

                if i == 100 or i == 300 or i == 370 * 100:
                    print('i:', i)
                    print('divisor:', np.nanmean(divisor))
                    print('backscatter_image[i, :]:', np.nanmean(
                        backscatter_image[i, :]))
                    print('profile_x_min:', np.nanmean(profile_x_min))
                    print('rslc[i, :] ** 2:', np.absolute(rslc[i, :]**2))

        del backscatter_image

        if self.min_range_index is not None:
            rslc[:, 0:self.min_range_index - 1] = np.nan

        if self.max_range_index is not None:
            rslc[:, self.max_range_index:-1] = np.nan

        del rslc_obj[pol_path]
        rslc_obj.create_dataset(pol_path, data=rslc)

        del rslc

    def get_metadata(self, freq, abs_cal_factor, pol_count, pol,
                     nlooks_y, nlooks_x, profile_filter_size):
        metadata_dict = {
            'REFERENCE_RSLC': self.input_file,
            'FREQUENCY': freq,
            'POLARIZATION': pol,
            'FLAG_IGNORE_NOISE': self.flag_ignore_noise,
            'PROFILE_FILTER_SIZE': profile_filter_size,
            'NLOOKS_Y': nlooks_y,
            'NLOOKS_X': nlooks_x
        }

        if abs_cal_factor is not None:
            metadata_dict['ABSOLUTE_CALIBRATION_FACTOR'] = \
                abs_cal_factor[pol_count]
        else:
            metadata_dict['ABSOLUTE_CALIBRATION_FACTOR'] = '(NOT SPECIFIED)'

        if freq == 'A':
            metadata_dict['RADIOMETRIC_TERRAIN_CONVENTION'] = \
                self.radiometric_terrain_convention_a
            metadata_dict['WATER_MASK'] = self.water_mask_a
            metadata_dict['TOPO_DIR'] = self.topo_dir_a
        else:
            metadata_dict['RADIOMETRIC_TERRAIN_CONVENTION'] = \
                self.radiometric_terrain_convention_b
            metadata_dict['WATER_MASK'] = self.water_mask_b
            metadata_dict['TOPO_DIR'] = self.topo_dir_b

        return metadata_dict

    def get_correction_profiles(
            self, profiles_directory,
            nisar_product_obj, freq, abs_cal_factor,
            pol_count, pol, backscatter_image,

            radar_grid_ml,

            profile_filter_size,
            metadata_dict,
            display_profiles_dict):

        slant_ranges_ml = radar_grid_ml.slant_ranges
        sensing_times_ml = radar_grid_ml.sensing_times

        mask_out_ind = None
        if self.minimum_azimuth_valid_percentage is not None:

            mask_out_ind = \
                (np.mean(~np.isnan(backscatter_image), axis=0) <
                 (float(self.minimum_azimuth_valid_percentage) / 100.0))

            mask_out_ind = binary_dilation(mask_out_ind, structure=np.ones(3))

            print('masking out azimuth lines with valid percentage below'
                  f' {self.minimum_azimuth_valid_percentage}%')
            print(f'number of valid azimuth lines:'
                  f' {np.count_nonzero(~mask_out_ind)} '
                  f'out of {mask_out_ind.size}')

        profile_x_max = np.nanpercentile(backscatter_image, self.percentile,
                                         axis=0)
        if mask_out_ind is not None:
            profile_x_max[mask_out_ind] = np.nan

        profile_x_stddev = np.nanstd(plant.get_db(backscatter_image), axis=0)
        if mask_out_ind is not None:
            profile_x_stddev[mask_out_ind] = np.nan

        if self.flag_ignore_noise:
            print('*** ignoring noise in profile calculation')
            profile_x_min = 0 * profile_x_max

        elif ((freq == 'A' and self.water_mask_a is not None) or
                (freq == 'B' and self.water_mask_b is not None)):
            print('*** using water mask to calculate profile_x_min for'
                  f' frequency {freq}')
            if freq == 'A':
                water_mask = self.read_image(self.water_mask_a).image
            else:
                water_mask = self.read_image(self.water_mask_b).image
            water_image = backscatter_image.copy()
            water_image[(water_mask != 80) &
                        (water_mask != 0)] = np.nan

            output_water_image = os.path.join(self.images_directory,
                                              f'water_image_{freq}_{pol}.tif')
            plant.save_image(water_image,
                             output_file=output_water_image,
                             force=True,
                             metadata=metadata_dict)

            profile_x_min = np.nanpercentile(water_image, 50, axis=0)
            del water_mask

        else:
            print('*** using noise equivalent backscatter to calculate'
                  f' profile_x_min for frequency {freq}')

            sensing_times_ml_cropped = sensing_times_ml

            noise_product = \
                nisar_product_obj.getResampledNoiseEquivalentBackscatter(
                    sensing_times=sensing_times_ml_cropped,
                    slant_ranges=slant_ranges_ml,
                    frequency=freq,
                    pol=pol)

            if abs_cal_factor is None:
                profile_x_min = np.nanmean(
                    noise_product.power_linear, axis=0)
            else:
                profile_x_min = np.nanmean(
                    (abs_cal_factor[pol_count] *
                     noise_product.power_linear), axis=0)

        if mask_out_ind is not None:
            profile_x_min[mask_out_ind] = np.nan

        if profile_filter_size > 0:
            print('applying profile filter of size',
                  profile_filter_size)

            shape = profile_x_max.shape
            profile_x_min = plant.filter_data(
                profile_x_min,
                mean=[1, profile_filter_size]).reshape(shape)
            profile_x_max = plant.filter_data(
                profile_x_max,
                mean=[1, profile_filter_size]).reshape(shape)
            profile_x_stddev = plant.filter_data(
                profile_x_stddev,
                mean=[1, profile_filter_size]).reshape(shape)

            if mask_out_ind is not None:
                profile_x_min[mask_out_ind] = np.nan
                profile_x_max[mask_out_ind] = np.nan
                profile_x_stddev[mask_out_ind] = np.nan

        print('*** profile_x_min.shape:', profile_x_min.shape)
        print('*** profile_x_max.shape:', profile_x_max.shape)

        profile_x_diff = profile_x_max - profile_x_min

        percentile_mean = np.nanmedian(profile_x_diff)

        output_profile_x_max = os.path.join(profiles_directory,
                                            'slant_range',
                                            f'profile_x_max_{freq}_{pol}.tif')
        output_profile_x_min = os.path.join(profiles_directory,
                                            'slant_range',
                                            f'profile_x_min_{freq}_{pol}.tif')
        output_profile_x_stddev = os.path.join(
            profiles_directory,
            'slant_range',
            f'profile_x_stddev_{freq}_{pol}.tif')
        output_divisor = os.path.join(profiles_directory,
                                      'slant_range',
                                      f'divisor_{freq}_{pol}.tif')
        plant.save_image(profile_x_max,
                         output_file=output_profile_x_max,
                         metadata=metadata_dict,
                         force=True)
        plant.save_image(profile_x_min,
                         output_file=output_profile_x_min,
                         metadata=metadata_dict,
                         force=True)
        plant.save_image(profile_x_stddev,
                         output_file=output_profile_x_stddev,
                         metadata=metadata_dict,
                         force=True)

        divisor = profile_x_diff / percentile_mean
        divisor_min_value = \
            np.power(10., self.divisor_min_value_db / 10.)

        print('divisor_min_value:', divisor_min_value)
        divisor = np.clip(divisor, divisor_min_value, None)

        plant.save_image(divisor,
                         output_file=output_divisor,
                         metadata=metadata_dict,
                         force=True)

        slant_range_file = os.path.join(profiles_directory,
                                        'slant_range',
                                        f'slant_range_{freq}.tif')
        plant.save_image(slant_ranges_ml,
                         output_file=slant_range_file,
                         metadata=metadata_dict,
                         force=True)

        if 'slant_ranges' not in display_profiles_dict.keys():
            display_profiles_dict['slant_ranges'] = slant_ranges_ml

        display_profiles_dict[pol] = plant.get_db(divisor)

        display_profiles_dict[pol + '_stddev'] = profile_x_stddev

        display_profiles_dict[pol + '_offset'] = plant.get_db(percentile_mean)

        return profile_x_min, divisor, profile_x_stddev

    def read_existing_profiles(self, profiles_directory, freq, pol,
                               slant_ranges):

        print('reading existing profiles for freq', freq, 'pol', pol)
        profile_x_min_file = os.path.join(profiles_directory,
                                          'slant_range',
                                          f'profile_x_min_{freq}_{pol}.tif')
        divisor_file = os.path.join(profiles_directory,
                                    'slant_range',
                                    f'divisor_{freq}_{pol}.tif')
        slant_range_file = os.path.join(profiles_directory,
                                        'slant_range',
                                        f'slant_range_{freq}.tif')
        profile_x_stddev_file = os.path.join(
            profiles_directory,
            'slant_range',
            f'profile_x_stddev_{freq}_{pol}.tif')

        profile_x_min_orig = plant.read_image(profile_x_min_file).image[0, :]
        profile_x_stddev_orig = plant.read_image(
            profile_x_stddev_file).image[0, :]
        divisor_orig = plant.read_image(divisor_file).image[0, :]
        slant_range_orig = plant.read_image(slant_range_file).image[0, :]

        if self.flag_ignore_noise:
            profile_x_min_orig = profile_x_min_orig * 0
            profile_x_min_orig[
                np.logical_not(np.isfinite(profile_x_min_orig))] = 0

        if np.array_equal(slant_ranges, slant_range_orig):
            return profile_x_min_orig, divisor_orig, profile_x_stddev_orig

        print('slant range from product and profile file do not match.')
        print('interpolating divisor and profile_x_min to match slant'
              ' range')

        print('*** slant_ranges.shape:', slant_ranges.shape)
        print('*** slant_range_orig.shape:',
              slant_range_orig.shape)
        print('*** divisor_orig.shape before:', divisor_orig.shape)
        print('*** profile_x_min_orig.shape before:',
              profile_x_min_orig.shape)

        profile_x_min_orig = profile_x_min_orig
        profile_x_stddev_orig = profile_x_stddev_orig
        divisor_orig = divisor_orig
        slant_range_orig = slant_range_orig

        f = interp1d(slant_range_orig, divisor_orig, kind='cubic')

        divisor = f(slant_ranges)
        divisor[divisor <= 0] = np.nan

        f = interp1d(slant_range_orig, profile_x_min_orig, kind='cubic')

        profile_x_min = f(slant_ranges)
        profile_x_min[profile_x_min <= 0] = 0

        f = interp1d(slant_range_orig, profile_x_stddev_orig, kind='cubic')

        profile_x_stddev = f(slant_ranges)

        return profile_x_min, divisor, profile_x_stddev


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        with PlantIsce3RslcEapAnalysis(parser, argv) as self_obj:
            ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
