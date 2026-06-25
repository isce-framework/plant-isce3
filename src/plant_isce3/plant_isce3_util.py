#!/usr/bin/env python3

import os
import shutil
import sys

import numpy as np
import plant
import plant_isce3
import datetime
import h5py
import isce3

import geopandas as gpd
from shapely.geometry import Polygon
from datetime import datetime

from plant_isce3.readers import open_product

POL_LIST = ['HH', 'HV', 'VH', 'VV', 'RH', 'RV', 'XX']

LIST_OF_STATIC_LAYERS = [
    'digitalElevationModel',
    'layoverShadowMask',
    'localIncidenceAngle',
    'losUnitVectorX',
    'losUnitVectorY',
    'waterMask',
    'rtcGammaToSigmaFactor',
    'rtcGammaToBetaFactor'
]


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=1,
                            cmap=1,
                            band=1,
                            mask=1,
                            default_output_options=1,
                            default_flags=1,
                            output_format=1,
                            multilook=1,
                            output_file=1,
                            output_dir=1)

    plant_isce3.add_arguments(parser,
                              burst_ids=1,
                              orbit_files=1,
                              nlooks_by_frequency=1,
                              frequency=1)

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--add-range-delay-in-meters',
                       dest='add_range_delay_m',
                       type=float,
                       help=('Add range delay in meters'))

    group.add_argument('--cp-pol',
                       '--copy-pol',
                       dest='copy_pol',
                       nargs=2,
                       type=str,
                       help=('Copy polarization. Provide source and'
                             ' destination pols.'))

    group.add_argument('--mv-pol',
                       '--move-pol',
                       '--rename-pol',
                       dest='move_pol',
                       nargs=2,
                       type=str,
                       help=('Rename polarization. Provide source and'
                             ' destination pols.'))

    group.add_argument('--swap-quad-pol',
                       dest='swap_quad_pol',
                       action='store_true',
                       help=("Swap quad-pol channels"))

    group.add_argument('--rm-pol',
                       '--remove-pol',
                       dest='remove_pol',
                       type=str,
                       help=('Remove polarization.'))

    group.add_argument('--runconfig',
                       '--runconfig-file',
                       dest='runconfig_file',
                       action='store_true',
                       help=("Extract the runconfig used to generate the"
                             " product from its metadata."))

    group.add_argument('--all-layers',
                       '--save-all-layers',
                       dest='flag_all_layers',
                       action='store_true',
                       help=('Save all layers (only available for NISAR L2'
                             ' products)'))

    group.add_argument('--all-secondary-layers',
                       '--save-all-secondary-layers',
                       dest='flag_all_secondary_layers',
                       action='store_true',
                       help=('Save all secondary layers (only available for'
                             ' NISAR L2 products)'))

    group.add_argument('--data',
                       '--images',
                       dest='data_file',
                       action='store_true',
                       help=("Extract product's imagery"))

    group.add_argument('--masked-data',
                       '--masked-images',
                       dest='masked_data_file',
                       action='store_true',
                       help=("Extract product's imagery and apply valid data"
                             " mask"))

    group.add_argument('--mask',
                       '--mask-layer',
                       dest='mask_file',
                       action='store_true',
                       help=("Extract mask layer."))

    group.add_argument('--layover-shadow-mask',
                       '--layover-shadow-mask-layer',
                       dest='layover_shadow_mask_file',
                       action='store_true',
                       help=("Extract the layover/shadow mask from the"
                             " product"))

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

    group.add_argument('--orbit-kml',
                       dest='orbit_kml_file',
                       action='store_true',
                       help=("Save a KML file containing the product's orbit"
                             " ephemeris"))

    group.add_argument('--footprint-html',
                       dest='footprint_html_file',
                       action='store_true',
                       help=("Save an HTML file containing the product's"
                             " footprint"))

    group.add_argument('--slant-range',
                       '--slant-range-file',
                       dest='slant_range_file',
                       action='store_true',
                       help=("Save file containing slant-range indexes."
                             " Only available for slant-range products"
                             " (level 1)"))

    group.add_argument('--azimuth-time',
                       '--azimuth-time-file',
                       dest='azimuth_time_file',
                       action='store_true',
                       help=("Save file containing azimuth times."
                             " Only available for slant-range products"
                             " (level 1)"))

    parser.add_argument('--no-bistatic-delay-correction',
                        dest='apply_bistatic_delay_correction',
                        default=True,
                        action='store_false',
                        help=("Prevent the bistatic delay to be applied"))

    parser.add_argument('--no-tropospheric-delay-correction',
                        dest='apply_static_tropospheric_delay_correction',
                        default=True,
                        action='store_false',
                        help=(""))

    parser.add_argument('--beta0',
                        dest='flag_output_complex',
                        default=True,
                        action='store_false',
                        help=("Prevent the static tropospheric delay to be"
                              " applied"))

    parser.add_argument('--no-thermal-correction',
                        dest='flag_thermal_correction',
                        default=True,
                        action='store_false',
                        help=("Prevent thermal noise correction to be applied"
                              ))

    parser.add_argument('--no-abs-rad-correction',
                        dest='flag_apply_abs_rad_correction',
                        default=True,
                        action='store_false',
                        help=(""))

    parser.add_argument('--prefix',
                        '--file-prefix',
                        dest='file_prefix',
                        type=str,
                        default='',
                        help="File prefix for option `--all-gcov-layers`")

    parser.add_argument('--square', '--sq',
                        action='store_true',
                        dest='transform_square',
                        help='Square of input')

    parser.add_argument('--separate-pol',
                        '--sep-pol',
                        '--separate-pols',
                        '--sep-pols',
                        '--separate-polarizations',
                        '--sep-polarizations',
                        dest='separate_pol',
                        action='store_true',
                        help='Handle polarimetric channels individually,'
                        ' with one output file for each'
                        ' available polarization. Requires the output'
                        ' directory argument: "--output-dir" or "--od"')

    parser.add_argument('--separate-freq',
                        '--sep-freq',
                        '--separate-freqs',
                        '--sep-freqs',
                        '--separate-frequencies',
                        '--sep-frequencies',
                        dest='separate_freq',
                        action='store_true',
                        help='Handle frequencies individually,'
                        ' with one output file for each'
                        ' available frequencies. Requires the output'
                        ' directory argument: "--output-dir" or "--od"')

    parser.add_argument('--zero-doppler-datetime-selection',
                        dest='zero_doppler_datetime_selection',
                        nargs=2,
                        type=str,
                        help='Zero-Doppler datetime selection')

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


class PlantIsce3Util(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        self.frequency_orig = self.frequency

        if not self.output_ext:
            self.output_ext = '.tif'

        if (self.separate_pol and not self.data_file and
                not self.masked_data_file):
            print('ERROR the option --separate-pol requires the option'
                  ' --data or --masked-data to be set')
            sys.exit(1)
            return

        if (self.separate_freq and
                (self.copy_pol or self.move_pol or
                 self.swap_quad_pol or self.remove_pol or
                 self.runconfig_file or
                 self.orbit_kml_file or
                 self.footprint_html_file or
                 (self.output_file and self.output_file.endswith('.h5')))):
            print('ERROR the option --separate-freq is'
                  ' not compatible with the options --copy-pol, --move-pol,'
                  ' --swap-pol, --remove-pol, --runconfig-file,'
                  ' --orbit-kml or with output files in h5 format')
            sys.exit(1)
            return

        plant_product_obj = self.load_product()
        if self.separate_pol or self.separate_freq:

            if not self.output_dir:
                self.output_dir = '.'

            if (plant_product_obj.sensor_name != 'NISAR'):
                self.print('ERROR the options --separate-pol and'
                           ' --separate-freq are only available for'
                           ' NISAR products')
                sys.exit(1)
                return

            freq_pols = plant_product_obj.nisar_product_obj.polarizations

            frequency_orig = self.frequency
            band_orig = self.band

            output_file_orig = self.output_file

            if self.separate_freq:
                freqs_iterator = freq_pols.items()
            else:
                frequency = plant_product_obj.get_frequency()
                pols_iterator = freq_pols[frequency]
                freqs_iterator = [[frequency, pols_iterator]]

            if output_file_orig:
                self.print(f'## output file template: {output_file_orig}')

            if not self.output_ext.startswith('.'):
                self.output_ext = f'.{self.output_ext}'

            ret_list = []
            for freq, pols in freqs_iterator:
                if (frequency_orig is not None and
                        frequency_orig != freq):
                    continue

                if not self.separate_pol:

                    if output_file_orig:
                        self.output_file = output_file_orig
                        self.output_file = self.output_file.replace(
                            '{frequency}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{freq}', self.frequency)
                    else:
                        self.output_file = os.path.join(
                            self.output_dir,
                            f'data_freq_{freq}{self.output_ext}')

                    self.frequency = freq

                    if (self.output_skip_if_existent and
                            plant.isfile(self.output_file)):
                        print('INFO output file %s already exist, '
                              'skipping execution..' % self.output_file)
                        continue

                    ret = self.run_util()
                    continue

                for band, pol in enumerate(pols):
                    if (band_orig is not None and
                            band_orig != band):
                        continue

                    if band_orig is not None and band != band_orig:
                        continue

                    self.frequency = freq
                    self.band = band

                    if output_file_orig:
                        self.output_file = output_file_orig
                        self.output_file = self.output_file.replace(
                            '{frequency}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{freq}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{polarization}', pol)
                        self.output_file = self.output_file.replace(
                            '{pol}', pol)
                    else:
                        self.output_file = os.path.join(
                            self.output_dir,
                            f'data_freq_{freq}_{pol}{self.output_ext}')

                    if (self.output_skip_if_existent and
                            plant.isfile(self.output_file)):
                        print('INFO output file %s already exist, '
                              'skipping execution..' % self.output_file)
                        continue
                    ret = self.run_util()

                ret_list.append(ret)
            return ret_list
        elif (not self.output_file and
              (self.output_dir or self.output_ext) and not
                (self.flag_all_layers or self.flag_all_secondary_layers)):
            self.parser.print_usage()
            self.print('ERROR this script only accepts --output-dir or'
                       ' --output-ext in --separate-pol,'
                       ' --separate-freq, --all-layers, and'
                       ' --all-secondary-layers modes')
            sys.exit(1)
            return
        return self.run_util()

    def run_util(self):

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        plant_product_obj = self.load_product()

        if (not self.input_file.endswith('.h5') and
                not self.input_file.endswith('.nc') and
                not self.input_file.endswith('.SAFE') and
                not self.input_file.endswith('.zip')):

            self.run_raster_as_input()
            plant.append_output_file(self.output_file)
            return self.output_file

        if self.orbit_kml_file:
            self.save_orbit_kml(plant_product_obj)

        elif self.footprint_html_file:
            self.save_footprint_html(plant_product_obj)

        elif self.slant_range_file:
            self.save_slant_range_file(plant_product_obj)

        elif self.azimuth_time_file:
            self.save_azimuth_time_file(plant_product_obj)

        elif (plant_product_obj.sensor_name == 'Sentinel-1'):
            self.run_sentinel_1_as_input(plant_product_obj)

        else:
            self.run_nisar_as_input(plant_product_obj)

        plant.append_output_file(self.output_file)
        return self.output_file

    def run_sentinel_1_as_input(self, plant_product_obj):

        flag_output_complex = self.flag_output_complex
        flag_thermal_correction = self.flag_thermal_correction
        flag_apply_abs_rad_correction = self.flag_apply_abs_rad_correction

        input_raster = plant_product_obj.get_sentinel_1_input_raster(
            flag_output_complex=flag_output_complex,
            flag_thermal_correction=flag_thermal_correction,
            flag_apply_abs_rad_correction=flag_apply_abs_rad_correction)

        image_obj = plant.read_image(input_raster)

        if self.mask_file:
            raise NotImplementedError
        elif self.layover_shadow_mask_file:
            raise NotImplementedError
        elif self.runconfig_file:
            raise NotImplementedError

        self.save_image_obj(image_obj)

    def get_rrsd_path(self, pol):
        path = f'/science/LSAR/RRSD/swaths/frequencyA/tx{pol[0]}/rx{pol[1]}/'
        return path

    def run_nisar_as_input(self, plant_product_obj):
        nisar_product_obj = open_product(self.input_file)

        if (self.masked_data_file and
                nisar_product_obj.productType not in ['GCOV', 'GSLC']):
            self.print('ERROR the option --masked-data is only available for'
                       ' NISAR GCOV and GSLC products')
            return

        if self.flag_all_layers or self.flag_all_secondary_layers:
            if nisar_product_obj.productType == 'STATIC':
                self.nlooks_az, self.nlooks_rg = self.get_nlooks()

                suffix = ''
                return self.save_all_layers(nisar_product_obj,
                                            plant_product_obj,
                                            suffix)

            if self.frequency is not None:

                self.nlooks_az, self.nlooks_rg = \
                    self.get_nlooks(self.frequency)

                suffix = f'_freq_{self.frequency.upper()}'

                return self.save_all_layers(nisar_product_obj,
                                            plant_product_obj,
                                            suffix)

            else:
                frequencies = nisar_product_obj.polarizations.keys()
                for freq in frequencies:
                    self.print(f'## processing frequency {freq}')
                    suffix = f'_freq_{freq.upper()}'
                    self.frequency = freq

                    self.nlooks_az, self.nlooks_rg = self.get_nlooks(freq)

                    self.save_all_layers(nisar_product_obj,
                                         plant_product_obj,
                                         suffix)
                self.frequency = None

            return

        if nisar_product_obj.productType == 'STATIC':
            freq_pol_dict = {None: None}
            if self.frequency is not None:
                print('WARNING the NISAR STATIC product does not'
                      ' contain frequency information; '
                      'ignoring the --frequency parameter.')
                self.frequency = None

        elif self.frequency is None:
            freq_pol_dict = nisar_product_obj.polarizations
            self.frequency = list(freq_pol_dict.keys())[0]
            print('WARNING frequency not specified, using first'
                  f' available frequency in the product: {self.frequency}')
        else:
            freq_pol_dict = {self.frequency:
                             nisar_product_obj.polarizations[self.frequency]}

        if self.zero_doppler_datetime_selection is not None:
            if self.select_row is not None:
                print('ERROR select row (--row/--rows) must be empty to use'
                      ' zero-Doppler selection')
                return

            zero_doppler_datetime_selection_list = \
                self.zero_doppler_datetime_selection

            kwargs_get_radar_grid = {}
            if self.frequency is not None:
                kwargs_get_radar_grid['frequency'] = self.frequency

            radar_grid = plant_product_obj.get_radar_grid(
                **kwargs_get_radar_grid)
            epoch = radar_grid.ref_epoch

            start_datetime = datetime.fromisoformat(
                zero_doppler_datetime_selection_list[0])
            end_datetime = datetime.fromisoformat(
                zero_doppler_datetime_selection_list[1])
            epoch_datetime = datetime.fromisoformat(
                epoch.isoformat())
            az_time_line_start = int(radar_grid.azimuth_index(
                (start_datetime - epoch_datetime).total_seconds()))
            az_time_line_end = int(radar_grid.azimuth_index(
                (end_datetime - epoch_datetime).total_seconds()))

            print('zero-Doppler datetime selection:')
            with plant.PlantIndent():
                print('Start datetime:', start_datetime)
                print('End datetime:', end_datetime)
                print('Epoch datetime:', epoch_datetime)
                print('Azimuth line start:', az_time_line_start)
                print('Azimuth line end:', az_time_line_end)

            self.select_row = \
                f'{az_time_line_start}:{az_time_line_end + 1}'
            self.plant_transform_obj.select_row = \
                f'{az_time_line_start}:{az_time_line_end + 1}'

        metadata_path = nisar_product_obj.MetadataPath
        if nisar_product_obj.productType == 'STATIC':
            pol_list = [None]
        else:
            pol_list = nisar_product_obj.polarizations[self.frequency]

        if self.output_ext:
            ext = self.output_ext
        else:
            ext = 'tif'

        if ext.startswith('.'):
            ext = ext[1:]

        if self.mask_file:
            self.save_mask(nisar_product_obj)

        elif self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(nisar_product_obj)

        elif self.rtc_gamma_to_sigma_file:

            self.save_nisar_layer('rtcGammaToSigmaFactor', nisar_product_obj)

        elif self.number_of_looks_file:

            self.save_nisar_layer('numberOfLooks', nisar_product_obj)

        elif self.data_file:
            self.save_data(plant_product_obj, nisar_product_obj)

        elif self.masked_data_file:
            self.save_data(plant_product_obj, nisar_product_obj,
                           masked=True)

        elif self.elevation_antenna_pattern_file:
            self.save_lut(
                f'{metadata_path}/calibrationInformation/'
                f'frequency{self.frequency}/'
                'elevationAntennaPattern/{pol}',
                pol_list=pol_list,
                force=True)

        elif self.noise_equivalent_backscatter_file:
            self.save_lut(
                f'{metadata_path}/calibrationInformation/'
                f'frequency{self.frequency}/'
                'noiseEquivalentBackscatter/{pol}',
                pol_list=pol_list,
                force=True)

        elif self.doppler_centroid_file:

            self.save_lut(
                f'{metadata_path}/processingInformation/'
                f'parameters/frequency{self.frequency}/'
                'dopplerCentroid',
                force=True)

        elif self.runconfig_file:
            self.save_runconfig_file(nisar_product_obj)

        elif (not self.output_file.endswith('.h5') and
                not self.output_file.endswith('.nc')):
            self.save_data(plant_product_obj, nisar_product_obj)
        else:
            if self.input_file != self.output_file:

                input_file_obj = plant.h5py_file_wrapper(self.input_file, 'r')
                input_file_obj.close()
                output_dir = os.path.dirname(self.output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                shutil.copyfile(self.input_file, self.output_file)

            product_type = nisar_product_obj.productType

            flag_apply_transformation = \
                self.plant_transform_obj.flag_apply_transformation()

            swaths_base_path = nisar_product_obj.SwathPath

            if (flag_apply_transformation and
                    nisar_product_obj.productType != 'RSLC'):
                self.print('ERROR the option to apply transformations is only'
                           ' available for NISAR RSLC products')
                return

            elif flag_apply_transformation:

                if (self.plant_transform_obj.flag_apply_crop() and
                        self.frequency_orig is None):
                    first_frequency = list(freq_pol_dict.keys())[0]
                    self.print('WARNING the option to crop an RSLC'
                               ' requires the frequency to be specified.'
                               ' Using the first available frequency:'
                               f' {first_frequency}')
                    self.frequency = first_frequency

                self.apply_transformations_rslc(freq_pol_dict,
                                                swaths_base_path,
                                                nisar_product_obj)

            if (self.add_range_delay_m and
                    nisar_product_obj.productType != 'RSLC'):
                self.print('ERROR the option to add range delay is only'
                           ' available for NISAR RSLC products')
                return

            elif self.add_range_delay_m:

                with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
                    for frequency in freq_pol_dict.keys():
                        image_group_path = (f'{swaths_base_path}/'
                                            f'frequency{frequency}')
                        slant_range_path = f'{image_group_path}/slantRange'
                        slant_range_vector = root_ds[slant_range_path][()]

                        del root_ds[slant_range_path]

                        print('frequency:', frequency)
                        print('slant_range_vector:', slant_range_vector)
                        print('self.add_range_delay_m:',
                              self.add_range_delay_m)

                        slant_range_vector += self.add_range_delay_m
                        print('slant_range_vector (updated):',
                              slant_range_vector)

                        root_ds.create_dataset(slant_range_path,
                                               data=slant_range_vector)

            if self.copy_pol:
                with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.copy_pol[0],
                                            output_pol=self.copy_pol[1],
                                            product_type=product_type)
                    print('done')

            if self.remove_pol:
                with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.remove_pol,
                                              product_type=product_type)
                    print('done')

            if self.move_pol:
                with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.move_pol[0],
                                            output_pol=self.move_pol[1],
                                            product_type=product_type)
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.move_pol[0],
                                              product_type=product_type)
                    print('done')

            if self.swap_quad_pol:
                exclude_string = 'frequencyB'
                with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
                    if nisar_product_obj.productType == 'RRSD':
                        print('*** input product is RRSD!')
                        hh_path = self.get_rrsd_path('HH')
                        hv_path = self.get_rrsd_path('HV')
                        vh_path = self.get_rrsd_path('VH')
                        vv_path = self.get_rrsd_path('VV')

                        hh = root_ds[f'{hh_path}/HH']
                        del root_ds[f'{hh_path}/HH']

                        vh = root_ds[f'{vh_path}/VH']
                        del root_ds[f'{vh_path}/VH']

                        root_ds[vh_path].create_dataset('VH', data=hh)
                        del hh
                        root_ds[hh_path].create_dataset('HH', data=vh)
                        del vh

                        hv = root_ds[f'{hv_path}/HV']
                        del root_ds[f'{hv_path}/HV']

                        vv = root_ds[f'{vv_path}/VV']
                        del root_ds[f'{vv_path}/VV']

                        root_ds[vv_path].create_dataset('VV', data=hv)
                        del hv
                        root_ds[hv_path].create_dataset('HV', data=vv)
                        del vv

                        tx_h_path = ('/science/LSAR/RRSD/swaths/'
                                     'frequencyA/txH')
                        tx_h_slant_range_path = ('/science/LSAR/RRSD/swaths/'
                                                 'frequencyA/txH/slantRange')
                        tx_v_path = ('/science/LSAR/RRSD/swaths/'
                                     'frequencyA/txV')
                        tx_v_slant_range_path = ('/science/LSAR/RRSD/swaths/'
                                                 'frequencyA/txV/slantRange')

                        tx_h_slant_range = root_ds[tx_h_slant_range_path]
                        tx_v_slant_range = root_ds[tx_v_slant_range_path]

                        del root_ds[tx_h_slant_range_path]
                        root_ds[tx_h_path].create_dataset(
                            'slantRange', data=tx_v_slant_range)

                        del root_ds[tx_v_slant_range_path]
                        root_ds[tx_v_path].create_dataset(
                            'slantRange', data=tx_h_slant_range)

                        return

                    self.print('swapping polarizations HH and VH in freq. A')
                    product_type = nisar_product_obj.productType
                    with plant.PlantIndent():
                        self.swap_pol_recursive(root_ds, key='/',
                                                pol_1='HH',
                                                pol_2='VH',
                                                exclude_string=exclude_string,
                                                product_type=product_type)
                    self.print('swapping polarizations HV and VV in freq. A')
                    with plant.PlantIndent():
                        self.swap_pol_recursive(root_ds, key='/',
                                                pol_1='HV',
                                                pol_2='VV',
                                                exclude_string=exclude_string,
                                                product_type=product_type)

            print(f'## file saved: {self.output_file}')

    def apply_transformations_rslc(self, freq_pol_dict, swaths_base_path,
                                   nisar_product_obj,
                                   flag_recursion_first_run=True,
                                   min_range=None,
                                   max_range=None):

        if flag_recursion_first_run:
            self.print('## applying transformations to the RSLC product')

        with plant.h5py_file_wrapper(self.output_file, 'a') as root_ds:
            if flag_recursion_first_run:
                zero_doppler_path = f'{swaths_base_path}/zeroDopplerTime'
                zero_doppler_vector = root_ds[zero_doppler_path][()]
                length = len(zero_doppler_vector)
            else:

                first_frequency = list(freq_pol_dict.keys())[0]
                first_pol = freq_pol_dict[first_frequency][0]
                image_path = (f'{swaths_base_path}/'
                              f'frequency{first_frequency}/{first_pol}')

                pol_ref = (f'HDF5:"{self.output_file}":'
                           f'/{image_path}')
                length = plant.read_image(
                    pol_ref, flag_no_messages=True).image.shape[0]

            with plant.PlantIndent():
                for frequency in freq_pol_dict.keys():
                    print('frequency:', frequency)

                    image_group_path = \
                        f'{swaths_base_path}/frequency{frequency}'
                    slant_range_path = f'{image_group_path}/slantRange'
                    slant_range_vector = root_ds[slant_range_path][()]
                    width = len(slant_range_vector)

                    y0 = self.plant_transform_obj._offset_y
                    if y0 is None:
                        y0 = 0

                    print(f'*** length: {length}')

                    new_length = self.plant_transform_obj.length
                    print(f'*** new_length: {new_length}')
                    if new_length is None:
                        new_length = length
                        print(f'*** new_length: {new_length}')

                    yf = min(y0 + new_length, length)
                    print(f'*** y0: {y0}, yf: {yf}')

                    if min_range is not None:

                        x0 = np.searchsorted(slant_range_vector, min_range,
                                             side='left')

                    else:
                        x0 = self.plant_transform_obj._offset_x
                        if x0 is None:
                            x0 = 0

                    if max_range is not None:

                        xf = np.searchsorted(slant_range_vector, max_range,
                                             side='right')

                        new_width = xf - x0 + 1

                    else:
                        new_width = self.plant_transform_obj.width
                        if new_width is None:
                            new_width = width

                        xf = min(x0 + new_width, width)

                    flag_crop_y = (y0 != 0 or yf != length)
                    flag_crop_x = (x0 != 0 or xf != width)

                    if flag_crop_x or flag_crop_y:
                        self.print('cropping parameters (indices):')
                        with plant.PlantIndent():
                            self.print(f'x0: {x0}, xf: {xf}')
                            self.print(f'y0: {y0}, yf: {yf}')
                        self.print(f'new length: {new_length}')
                        self.print(f'new width: {new_width}')

                    if flag_crop_y and flag_recursion_first_run:
                        new_zero_doppler_vector = \
                            zero_doppler_vector[y0: yf]

                        self.update_h5_data(root_ds, zero_doppler_path,
                                            new_zero_doppler_vector)

                    if flag_crop_x:
                        image_group_path = (f'{swaths_base_path}/'
                                            f'frequency{frequency}')
                        slant_range_path = f'{image_group_path}/slantRange'
                        slant_range_vector = root_ds[slant_range_path][()]
                        new_slant_range_vector = slant_range_vector[x0: xf]
                        min_range = new_slant_range_vector[0]
                        max_range = new_slant_range_vector[-1]

                        self.update_h5_data(root_ds, slant_range_path,
                                            new_slant_range_vector)

                    for pol in freq_pol_dict[frequency]:
                        with plant.PlantIndent():
                            print('polarization:', pol)
                            image_path = \
                                (f'{swaths_base_path}/'
                                    f'frequency{frequency}/{pol}')

                            pol_ref = (f'HDF5:"{self.output_file}":'
                                       f'/{image_path}')

                            if (flag_crop_x and (min_range is not None or
                                                 max_range is not None)):
                                offset_x_orig = \
                                    self.plant_transform_obj._offset_x
                                width_orig = \
                                    self.plant_transform_obj._width
                                select_col_orig = \
                                    self.plant_transform_obj._select_col

                                self.plant_transform_obj._offset_x = x0
                                self.plant_transform_obj._width = new_width
                                self.plant_transform_obj._select_col = \
                                    f'{x0}:{xf}'
                            new_image_data = \
                                self.read_image(
                                    pol_ref, flag_no_messages=True).image

                            if (flag_crop_x and (min_range is not None or
                                                 max_range is not None)):
                                self.plant_transform_obj._offset_x = \
                                    offset_x_orig
                                self.plant_transform_obj._width = \
                                    width_orig
                                self.plant_transform_obj._select_col = \
                                    select_col_orig

                            self.update_h5_data(root_ds, image_path,
                                                new_image_data)

                freq_pol_orig_dict = nisar_product_obj.polarizations

                freq_pol_str = ', '.join(list(freq_pol_dict.keys()))
                if (flag_recursion_first_run and flag_crop_y and
                    set(freq_pol_dict.keys()) !=
                        set(nisar_product_obj.polarizations.keys())):
                    processed_freqs_str = ', '.join(list(freq_pol_dict.keys()))
                    remaining_frequencies = \
                        set(nisar_product_obj.polarizations.keys()) - \
                        set(freq_pol_dict.keys())
                    remaining_frequencies_str = ', '.join(list(
                        remaining_frequencies))

                    self.print('Cropping frequency'
                               f' {remaining_frequencies_str} images'
                               ' with cropping indices derived from'
                               f' the frequency {processed_freqs_str}')
                    for frequency in remaining_frequencies:

                        remaining_freq_pol_list = freq_pol_orig_dict[
                            frequency]

                        new_freq_pol_dict = {frequency:
                                             remaining_freq_pol_list}

                        self.apply_transformations_rslc(
                            new_freq_pol_dict, swaths_base_path,
                            nisar_product_obj,
                            flag_recursion_first_run=False,
                            min_range=min_range,
                            max_range=max_range)

                if flag_crop_y:
                    print('updating valid-samples subswath arrays'
                          ' in the azimuth direction'
                          f' for frequency: {freq_pol_str}')
                    for frequency in freq_pol_dict.keys():
                        for subswath_count in range(1, 6):
                            valid_samples_path = (
                                f'{swaths_base_path}/frequency{frequency}/'
                                f'validSamplesSubSwath{subswath_count}')
                            if valid_samples_path not in root_ds:
                                continue
                            valid_samples_array = \
                                root_ds[valid_samples_path][()]
                            print('*** valid_samples_array.shape (before): '
                                  f'{valid_samples_array.shape}')
                            new_valid_samples_array = \
                                valid_samples_array[y0: yf, :]
                            print('*** valid_samples_array.shape (after): '
                                  f'{new_valid_samples_array.shape}')
                            self.update_h5_data(root_ds, valid_samples_path,
                                                new_valid_samples_array)

                if flag_crop_x:

                    new_xf = xf - x0
                    print('updating valid-samples subswath arrays'
                          ' in the range direction clipping to new range'
                          f' limit ({new_xf}) for frequency: {freq_pol_str}')
                    for frequency in freq_pol_dict.keys():
                        for subswath_count in range(1, 6):
                            valid_samples_length, _ = \
                                new_valid_samples_array.shape
                            valid_samples_path = (
                                f'{swaths_base_path}/frequency{frequency}/'
                                f'validSamplesSubSwath{subswath_count}')

                            if valid_samples_path not in root_ds:
                                continue

                            valid_samples_array = \
                                root_ds[valid_samples_path][()]

                            for row in range(valid_samples_length):
                                x_min, x_max = valid_samples_array[row, :] - x0

                                if (x_min > x_max and x_min <= new_xf and
                                        x_max >= 0):

                                    print(f'row {row}: no valid samples')
                                    continue
                                elif x_min > x_max:

                                    valid_samples_array[row, :] = [1, 0]
                                    continue

                                valid_samples_array[row, :] = \
                                    np.clip(valid_samples_array[row, :],
                                            0, new_xf)

                            self.update_h5_data(root_ds, valid_samples_path,
                                                valid_samples_array)

    def update_h5_data(self, root_ds, image_path, new_image_data):
        new_dataset = root_ds.create_dataset(
            image_path + '_tpm', data=new_image_data)
        new_dataset.attrs.update(
            root_ds[image_path].attrs)
        del root_ds[image_path]
        root_ds.move(image_path + '_tpm', image_path)

    def save_all_layers(self, nisar_product_obj, plant_product_obj,
                        suffix=''):
        if not self.output_dir:
            self.print('ERROR this option requires the output'
                       ' directory (`--od / --output-dir).')
            return
        prefix = self.file_prefix

        if self.output_ext:
            ext = self.output_ext
        else:
            ext = 'tif'

        if ext.startswith('.'):
            ext = ext[1:]

        self.output_file = os.path.join(self.output_dir,
                                        f'{prefix}mask{suffix}.{ext}')

        if nisar_product_obj.productType == 'STATIC':
            for static_layer in LIST_OF_STATIC_LAYERS:

                self.output_file = os.path.join(
                    self.output_dir,
                    f'{prefix}{static_layer}{suffix}.{ext}')

                if static_layer == 'layoverShadowMask':

                    self.save_layover_shadow_mask(
                        nisar_product_obj=nisar_product_obj)
                    continue
                if static_layer == 'waterMask':

                    self.save_binary_water_mask(
                        nisar_product_obj=nisar_product_obj)
                    continue

                self.save_nisar_layer(static_layer, nisar_product_obj)

            return

        self.save_mask(nisar_product_obj)

        if nisar_product_obj.productType == 'GCOV':
            self.output_file = os.path.join(
                self.output_dir,
                f'{prefix}rtcGammaToSigmaFactor{suffix}.{ext}')
            self.save_nisar_layer('rtcGammaToSigmaFactor', nisar_product_obj)

            self.output_file = os.path.join(
                self.output_dir,
                f'{prefix}numberOfLooks{suffix}.{ext}')
            self.save_nisar_layer('numberOfLooks', nisar_product_obj)
            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}data{suffix}.{ext}')

        metadata_path = nisar_product_obj.MetadataPath
        pol_list = nisar_product_obj.polarizations[self.frequency]

        self.output_file = os.path.join(self.output_dir,
                                        f'{prefix}elevationAntennaPattern_'
                                        '{pol}' + f'{suffix}.{ext}')
        self.save_lut(f'{metadata_path}/calibrationInformation/'
                      f'frequency{self.frequency}/'
                      'elevationAntennaPattern/{pol}', pol_list=pol_list)

        self.output_file = os.path.join(
            self.output_dir, f'{prefix}noiseEquivalent'
            'Backscatter_{pol}' + f'{suffix}.{ext}')
        self.save_lut(f'{metadata_path}/calibrationInformation/'
                      f'frequency{self.frequency}/'
                      'noiseEquivalentBackscatter/{pol}',
                      pol_list=pol_list)

        self.output_file = os.path.join(
            self.output_dir,
            f'{prefix}dopplerCentroid{suffix}.{ext}')
        self.save_lut(f'{metadata_path}/processingInformation/'
                      f'parameters/frequency{self.frequency}/'
                      'dopplerCentroid')

        if nisar_product_obj.productType == 'GCOV':

            self.output_file = os.path.join(
                self.output_dir,
                f'{prefix}azimuthIonosphere{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/processingInformation/'
                          f'timingCorrections/frequency{self.frequency}/'
                          'azimuthIonosphere')

            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}slantRangeIonosphere'
                                            f'{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/processingInformation/'
                          f'timingCorrections/frequency{self.frequency}/'
                          'slantRangeIonosphere')

            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}rxHorizontalCrosspol'
                                            f'{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/calibrationInformation/'
                          'crosstalk/rxHorizontalCrosspol')

            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}txHorizontalCrosspol'
                                            f'{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/calibrationInformation/'
                          'crosstalk/txHorizontalCrosspol')

            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}rxVerticalCrosspol'
                                            f'{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/calibrationInformation/'
                          'crosstalk/rxVerticalCrosspol')

            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}txVerticalCrosspol'
                                            f'{suffix}.{ext}')
            self.save_lut(f'{metadata_path}/calibrationInformation/'
                          'crosstalk/txVerticalCrosspol')

        self.output_file = os.path.join(self.output_dir,
                                        f'{prefix}referenceTerrainHeight'
                                        f'{suffix}.{ext}')
        self.save_lut(f'{metadata_path}/processingInformation/'
                      'parameters/referenceTerrainHeight')

        if self.flag_all_layers:
            self.output_file = os.path.join(self.output_dir,
                                            f'{prefix}data'
                                            f'{suffix}.{ext}')
            self.save_data(plant_product_obj, nisar_product_obj)

    def run_raster_as_input(self):
        image_obj = plant.read_image(self.input_file)

        if self.mask_file:
            self.save_mask(image_obj=image_obj)

        elif self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(image_obj=image_obj)

        elif self.data_file:
            self.save_image_obj(image_obj)

        else:
            self.save_image(image_obj, self.output_file)

    def get_az_rg_timing_correction_luts(self, plant_product_obj, radar_grid):
        rg_step_meters = 10 * radar_grid.range_pixel_spacing
        az_step_meters = rg_step_meters
        apply_bistatic_delay_correction = \
            self.apply_bistatic_delay_correction
        apply_static_tropospheric_delay_correction = \
            self.apply_static_tropospheric_delay_correction

        dem_raster = plant_isce3.get_isce3_raster(self.dem_file)

        rg_lut, az_lut = plant_isce3.compute_correction_lut(
            plant_product_obj.burst,
            dem_raster,

            rg_step_meters,
            az_step_meters,
            apply_bistatic_delay_correction,
            apply_static_tropospheric_delay_correction)

        return rg_lut, az_lut

    def save_slant_range_file(self, plant_product_obj):

        radar_grid = plant_product_obj.get_radar_grid()
        new_var_array = np.repeat(
            [radar_grid.slant_ranges], radar_grid.length, axis=0)

        if plant_product_obj.sensor_name == 'Sentinel-1':
            rg_lut, _ = self.get_az_rg_timing_correction_luts(
                plant_product_obj, radar_grid)
        else:
            rg_lut = None

        if rg_lut is not None:
            rg_lut.bounds_error = False

            for i in range(radar_grid.length):
                range_shift = \
                    rg_lut.eval(radar_grid.sensing_times[i],
                                radar_grid.slant_ranges)

                new_var_array[i, :] = new_var_array[i, :] + range_shift

        plant_image_obj = plant.PlantImage(new_var_array)
        plant_image_obj.set_name('Slant-range Distance in Meters')
        self.save_image(plant_image_obj, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_azimuth_time_file(self, plant_product_obj):
        radar_grid = plant_product_obj.get_radar_grid()
        new_var_array = np.repeat(np.transpose(
            [radar_grid.sensing_times]),
            radar_grid.width, axis=1)

        if plant_product_obj.sensor_name == 'Sentinel-1':
            _, az_lut = self.get_az_rg_timing_correction_luts(
                plant_product_obj, radar_grid)
        else:
            az_lut = None

        if az_lut is not None:
            az_lut.bounds_error = False

            for i in range(radar_grid.length):
                for j in range(radar_grid.width):
                    azimuth_delay = \
                        az_lut.eval(radar_grid.sensing_times[i],
                                    radar_grid.slant_ranges[j])

                    new_var_array[i, j] = new_var_array[i, j] + azimuth_delay

        plant_image_obj = plant.PlantImage(new_var_array)

        ref_epoch = str(radar_grid.ref_epoch)
        plant_image_obj.set_name(f'Azimuth Time in Seconds Since {ref_epoch}')
        plant_image_obj.set_metadata(f'REF_EPOCH: {ref_epoch}[s]')

        self.save_image(plant_image_obj, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_image_obj(self, image_obj):
        if ('complex' not in plant.get_dtype_name(image_obj.dtype).lower() or
                self.flag_output_complex is not False):
            self.save_image(image_obj, output_file=self.output_file)
            plant.append_output_file(self.output_file)
            return

        image_list = []
        for b in range(image_obj.nbands):
            image_list.append(
                np.absolute(image_obj.get_image(band=b)) ** 2)

        self.save_image(image_list, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_data(self, plant_product_obj, nisar_product_obj,
                  masked=False):

        if (not self.transform_square and
                nisar_product_obj.productType == 'GSLC' and
                ((self.nlooks_az is not None and self.nlooks_az != 1) or
                 (self.nlooks_rg is not None and self.nlooks_rg != 1))):
            self.print('WARNING multilooking is enabled and'
                       ' GSLC samples are complex.'
                       ' Converting samples to power/intensity'
                       ' before multilooking.')
            self.transform_square = True

        input_raster = self.get_input_raster_from_nisar_product(
            input_raster=None,
            plant_product_obj=plant_product_obj)

        if (masked and
                plant_product_obj.nisar_product_obj.productType not
                in ['GCOV', 'GSLC']):
            self.print('WARNING the option --masked-data is only available for'
                       ' GCOV and GSLC products, skipping mask application..')
            masked = False

        if masked:
            image_ref = self.get_grids_ref(
                'mask', self.frequency, nisar_product_obj, image_obj=None,
                valid_products=['GCOV', 'GSLC'])

            image_obj = self.read_image(image_ref, band=0)

            mask_array_obj = plant.filter_data(image_obj,
                                               snap_to_multilook_grid=True,
                                               null=255,
                                               nlooks=[self.nlooks_az,
                                                       self.nlooks_rg])
            del image_obj

            mask_array_obj.image = np.asarray(mask_array_obj.image,
                                              dtype=np.uint8)

            valid = ((mask_array_obj.image > 0) & (mask_array_obj.image < 10))

            image_obj = plant.util(input_raster, input_mask=valid)

            self.save_image(image_obj, output_file=self.output_file,
                            force=True)

        else:
            self.save_image(input_raster, output_file=self.output_file,
                            force=True)

        plant.append_output_file(self.output_file)

    def save_mask(self, nisar_product_obj=None,
                  image_obj=None):

        image_ref = self.get_grids_ref(
            'mask', self.frequency, nisar_product_obj, image_obj,
            valid_products=['GCOV', 'GSLC'])

        image_obj = self.read_image(image_ref, band=0)

        mask_array_obj = plant.filter_data(
            image_obj,
            snap_to_multilook_grid=True,
            null=255,

            nlooks=[self.nlooks_az, self.nlooks_rg])
        del image_obj

        mask_array_obj.image = np.asarray(mask_array_obj.image,
                                          dtype=np.uint8)
        mask_array_obj.dtype = np.uint8

        mask_ctable = self.get_mask_ctable(mask_array_obj.image)

        self.save_image(mask_array_obj, output_file=self.output_file,
                        out_null=255, ctable=mask_ctable)
        plant.append_output_file(self.output_file)

    def save_nisar_layer(self, layer_name, nisar_product_obj=None,

                         image_obj=None):

        image_ref = self.get_grids_ref(
            layer_name, self.frequency, nisar_product_obj, image_obj)

        if self.nlooks_az != 1 or self.nlooks_rg != 1:

            temp_file = plant.get_temporary_file(ext='.tif', append=True)
            plant_isce3.multilook_isce3(image_ref,
                                        output_file=temp_file,
                                        output_format=self.output_format,
                                        nlooks_y=self.nlooks_az,
                                        nlooks_x=self.nlooks_rg)

            self.save_image(temp_file, output_file=self.output_file)
        else:
            self.save_image(image_ref, output_file=self.output_file,
                            force=True)

        plant.append_output_file(self.output_file)

    def save_layover_shadow_mask(self, nisar_product_obj=None,
                                 image_obj=None):

        image_obj = self.get_grids_ref(
            'layoverShadowMask', self.frequency, nisar_product_obj, image_obj)

        layover_shadow_mask_ctable = self.get_layover_shadow_mask_ctable()
        if self.nlooks_az != 1 or self.nlooks_rg != 1:

            image_obj = plant.filter(image_obj, output_file=self.output_file,

                                     nlooks=[self.nlooks_az, self.nlooks_rg],
                                     null=255,
                                     force=self.force)

        self.save_image(image_obj, output_file=self.output_file,
                        out_null=255, ctable=layover_shadow_mask_ctable)
        plant.append_output_file(self.output_file)

    def save_binary_water_mask(self, nisar_product_obj=None,
                               image_obj=None):

        image_obj = self.get_grids_ref('waterMask', self.frequency,
                                       nisar_product_obj, image_obj)

        binary_water_mask_ctable = self.get_binary_water_mask_ctable()
        if self.nlooks_az != 1 or self.nlooks_rg != 1:

            image_obj = plant.filter(
                image_obj, output_file=self.output_file,

                nlooks=[self.nlooks_az, self.nlooks_rg],
                null=255,
                force=self.force)

        self.save_image(image_obj, output_file=self.output_file,
                        out_null=255, ctable=binary_water_mask_ctable)
        plant.append_output_file(self.output_file)

    def save_lut(self, h5_path, pol_list=[], flag_skip_if_error=True,
                 force=None):
        print('*** save_lut() called with pol_list:', pol_list)
        if '{pol}' in h5_path:
            output_file_orig = self.output_file
            for pol in pol_list:
                self.output_file = output_file_orig.replace('{pol}', pol)
                print('*** saving LUT:', h5_path.replace('{pol}', pol))
                self.save_lut(h5_path.replace('{pol}', pol), force=force)
            self.output_file = output_file_orig
            return

        if force is None:
            force = self.force

        ref = f'NETCDF:{self.input_file}:{h5_path}'
        try:
            image_obj = plant.read_image(ref)
            self.save_image(image_obj, output_file=self.output_file,
                            force=force)
            plant.append_output_file(self.output_file)
        except BaseException:
            if flag_skip_if_error:
                return
            self.print(f'ERROR fail to open dataset "{h5_path}" from '
                       f' "{self.input_file}"')

    def save_runconfig_file(self, nisar_product_obj):

        h5_obj = plant.h5py_file_wrapper(self.input_file, 'r')

        runconfig_path = (f'/science/LSAR/{nisar_product_obj.productType}/'
                          'metadata/processingInformation/'
                          'parameters/runConfigurationContents')

        runconfig_str = str(h5_obj[runconfig_path][()].decode('utf-8'))
        h5_obj.close()
        runconfig_str = runconfig_str.replace("\\n", "\n") + '\n'

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, "w") as f:

            f.write(runconfig_str)
            f.close()

        print(f'## file saved: {self.output_file} (YAML)')
        plant.append_output_file(self.output_file)

    def save_orbit_kml(self, plant_product_obj):

        orbit = plant_product_obj.get_orbit()
        flag_has_polygon = False

        if plant_product_obj.sensor_name == 'NISAR':

            flag_has_polygon = True

            with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
                polygon = plant_isce3.get_nisar_product_bounding_polygon(
                    h5_obj, flag_as_list=True)

        ellipsoid = isce3.core.Ellipsoid()
        time_list = []
        llh_list = []
        state_vectors_pos = orbit.position
        state_vectors_vel = orbit.velocity
        state_vectors_time = orbit.time
        reference_epoch = orbit.reference_epoch

        with plant.PlantIndent():
            if flag_has_polygon:
                print('polygon: ', polygon)
            print('reference epoch:', reference_epoch)

        for pos, time in zip(state_vectors_pos, state_vectors_time):
            time_str = str(reference_epoch + isce3.core.TimeDelta(time))
            llh_list.append(ellipsoid.xyz_to_lon_lat(pos))
            time_list.append(time_str)

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, 'w') as fp:

            fp.write('<?xml version="1.0" encoding="UTF-8"?> \n')
            fp.write('<kml xmlns="http://www.opengis.net/kml/2.2" ')
            fp.write('xmlns:gx="http://www.google.com/kml/ext/2.2"> \n')
            fp.write('<Document> \n')

            if flag_has_polygon:
                self.add_polygon(fp, polygon)

            self.add_line(fp, state_vectors_pos, state_vectors_vel, time_list,
                          llh_list,
                          flag_altitude=False, color='#ff000000')
            self.add_line(fp, state_vectors_pos, state_vectors_vel, time_list,
                          llh_list,
                          flag_altitude=True, color='#ff00ffff')
            fp.write('<PolyStyle>')
            fp.write('  <color>7f0000ff</color>')
            fp.write('  <colorMode>normal</colorMode>')
            fp.write('  <fill>1</fill>')
            fp.write('  <outline>1</outline>')
            fp.write('</PolyStyle>')
            fp.write('<Schema id="schema">\n')
            fp.write('  <gx:SimpleArrayField name="UTC time" type="string">\n')
            fp.write('    <displayName>X</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="X" type="float">\n')
            fp.write('    <displayName>X</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="Y" type="float">\n')
            fp.write('    <displayName>Y</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="Z" type="float">\n')
            fp.write('    <displayName>Z</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('</Schema>\n')
            fp.write('</Document> \n')
            fp.write('</kml>\n')
        if plant.isfile(self.output_file) and self.verbose:
            print('## file saved: %s (KML)' % self.output_file)
        plant.append_output_file(self.output_file)

    def add_polygon(self, fp, polygon):
        fp.write('<Placemark>\n')
        fp.write('    <name>Swath Polygon</name>\n')
        fp.write('    <visibility>0</visibility>\n')
        fp.write('    <Polygon>\n')
        fp.write('      <extrude>1</extrude>\n')
        fp.write('      <altitudeMode>relativeToGround</altitudeMode>\n')
        fp.write('      <outerBoundaryIs>\n')
        fp.write('        <LinearRing>\n')
        fp.write('          <coordinates>\n')

        for vertex in polygon:
            if len(vertex) == 3:
                lon, lat, height = vertex
            else:
                lon, lat = vertex
                height = 0
            fp.write(f'            {lon},{lat},{height} \n')
        fp.write('          </coordinates>\n')
        fp.write('        </LinearRing>\n')
        fp.write('      </outerBoundaryIs>\n')
        fp.write('    </Polygon>\n')
        fp.write('  </Placemark>\n')

    def save_footprint_html(self, plant_product_obj):

        if plant_product_obj.sensor_name != 'NISAR':
            print('ERROR footprint HTML file can only be saved for NISAR'
                  ' products.')
            return

        with plant.h5py_file_wrapper(self.input_file, 'r') as h5_obj:
            coords = plant_isce3.get_nisar_product_bounding_polygon(
                h5_obj, flag_as_list=True)

        polygon = Polygon(coords)

        gdf = gpd.GeoDataFrame(
            {"name": ["My Polygon"]},
            geometry=[polygon],
            crs="EPSG:4326"
        )

        m = gdf.explore()
        m.save(self.output_file)

        if plant.isfile(self.output_file) and self.verbose:
            print(f'## file saved: {self.output_file}')

    def add_line(self, fp, state_vectors_pos, state_vectors_vel,
                 time_list,
                 llh_list, flag_altitude=True, color=None):
        fp.write('<Folder> \n')

        if flag_altitude:
            fp.write('<name>Platform Track</name>\n')
        else:
            fp.write('<name>Platform Ground Track</name>\n')

        fp.write('<kml:visibility>0</kml:visibility>')
        fp.write('<visibility>0</visibility>')

        for i, (llh, time_str) in enumerate(zip(llh_list, time_list)):

            fp.write('<Placemark>\n')
            if flag_altitude:
                fp.write(f'  <name>t{i}</name>\n')
            else:
                fp.write(f'  <name>g{i}</name>\n')

            description = f'UTC time: {time_str} \n'
            for axis, coord_str in enumerate(['X', 'Y', 'Z']):
                description += (f'    pos. {coord_str}:'
                                f' {state_vectors_pos[i][axis]}\n')
            for axis, coord_str in enumerate(['X', 'Y', 'Z']):
                description += (f'    vel. {coord_str}:'
                                f' {state_vectors_vel[i][axis]}\n')
            fp.write(f'  <description>{description}</description>\n')
            fp.write('  <Point>\n')

            lon = np.degrees(llh[0])
            lat = np.degrees(llh[1])
            h = llh[2] if flag_altitude else 0
            fp.write(f'   <coordinates> {lon}, {lat}, {h}</coordinates> \n')
            if flag_altitude:
                fp.write('   <altitudeMode>absolute</altitudeMode>\n')
            fp.write('  </Point>\n')
            fp.write('</Placemark>\n')

        fp.write('</Folder><Folder> \n')
        if flag_altitude:
            fp.write('<name>Platform Track</name>\n')
        else:
            fp.write('<name>Platform Ground Track</name>\n')

        fp.write('<Placemark> \n')
        if flag_altitude:
            fp.write('<name>Platform Track</name>')
        else:
            fp.write('<name>Platform Ground Track</name>')

        fp.write('<gx:Track> \n')

        for time_str in time_list:
            fp.write(f'  <when>{time_str}</when>\n')

        for llh in llh_list:
            lon = np.degrees(llh[0])
            lat = np.degrees(llh[1])
            h = llh[2] if flag_altitude else 0
            fp.write(f'   <gx:coord> {lon},{lat},{h}</gx:coord> \n')

        if flag_altitude:
            fp.write('   <altitudeMode>absolute</altitudeMode>\n')

        fp.write('<ExtendedData>\n')
        fp.write('<SchemaData schemaUrl="#schema">\n')

        fp.write('<gx:SimpleArrayData name="UTC time" kml:name="string">\n')
        for time_str in time_list:
            fp.write(f'<gx:value>{time_str}</gx:value>\n')
        fp.write('</gx:SimpleArrayData>\n')

        for e, coord_str in enumerate(['X', 'Y', 'Z']):
            fp.write(
                f'<gx:SimpleArrayData name="{coord_str}" kml:name="float">\n')
            for i, time_str in enumerate(time_list):
                fp.write(f'<gx:value>{state_vectors_pos[i][e]}</gx:value>\n')
            fp.write('</gx:SimpleArrayData>\n')

        fp.write('</SchemaData>\n')
        fp.write('</ExtendedData>\n')
        fp.write('</gx:Track>\n')
        fp.write('</Placemark> \n')
        fp.write('</Folder> \n')

    def move_pol_recursive(self, hdf_obj: h5py.Group, key: str, input_pol: str,
                           output_pol: str, exclude_string: str = '',
                           product_type: str = ''):
        self.print(f'copying pol "{input_pol}" to pol "{output_pol}"')
        with plant.PlantIndent():
            self.copy_pol_recursive(hdf_obj, key, input_pol, output_pol,
                                    exclude_string, product_type)
        self.print(f'removing pol "{input_pol}"')
        with plant.PlantIndent():
            self.remove_pol_recursive(hdf_obj, key, input_pol, exclude_string,
                                      product_type)

    def swap_pol_recursive(self, hdf_obj: h5py.Group, key: str, pol_1: str,
                           pol_2: str, exclude_string: str = '',
                           product_type: str = ''):
        self.print(f'moving pol "{pol_1}" to temporary pol "XX"')
        with plant.PlantIndent():
            self.move_pol_recursive(hdf_obj, key, pol_1, 'XX', exclude_string,
                                    product_type)
        self.print(f'moving pol "{pol_2}" to pol "{pol_1}"')
        with plant.PlantIndent():
            self.move_pol_recursive(hdf_obj, key, pol_2, pol_1, exclude_string,
                                    product_type)
        self.print(f'moving temporary pol "XX" to pol "{pol_2}"')
        with plant.PlantIndent():
            self.move_pol_recursive(hdf_obj, key, 'XX', pol_2, exclude_string,
                                    product_type)

    def copy_pol_recursive(self, hdf_obj: h5py.Group, key: str, input_pol: str,
                           output_pol: str, exclude_string: str = '',
                           product_type: str = ''):

        h5_element = hdf_obj[key]

        if key == input_pol:
            input_path = f'{hdf_obj.name}/{input_pol}'
            output_path = f'{hdf_obj.name}/{output_pol}'
            if output_path in hdf_obj:

                if isinstance(h5_element, h5py.Group):
                    element_str = 'H5 group'
                else:
                    element_str = 'H5 dataset'

                ret = overwrite_dataset_check(output_path, force=self.force,
                                              element_str=element_str)

                if ret is False:
                    return

                del hdf_obj[output_pol]

            print(f'copying" "{input_path}"')
            print(f'     to" "{output_path}"')
            hdf_obj.copy(input_path, hdf_obj, output_path)
            return

        if isinstance(h5_element, h5py.Group):

            for sub_key in h5_element.keys():
                if exclude_string and exclude_string in sub_key:
                    self.print(f'*** skipping: {exclude_string}')
                    continue
                self.copy_pol_recursive(h5_element, sub_key, input_pol,
                                        output_pol, exclude_string,
                                        product_type)
                if product_type == 'GCOV':

                    self.copy_pol_recursive(h5_element, sub_key,
                                            f'{input_pol}{input_pol}',
                                            f'{output_pol}{output_pol}',
                                            exclude_string,
                                            product_type='')

        elif input_pol in POL_LIST and key == 'listOfPolarizations':
            list_of_polarizations = h5_element[()]
            self.print(f'*** H5 path: {h5_element.name}')
            print('input list_of_polarizations:', list_of_polarizations)
            flag_input_pol_found = any([pol.decode() == input_pol
                                        for pol in list_of_polarizations])

            if not flag_input_pol_found:
                print(f'WARNING input pol {input_pol} not found in the list'
                      f' of polarization at: "{h5_element.name}"')
                return
            flag_output_pol_found = any([pol.decode() == output_pol
                                        for pol in list_of_polarizations])

            if flag_output_pol_found:
                print(f'WARNING output pol {output_pol} already exists in list'
                      f' of polarization at: "{h5_element.name}"')
                return
            list_of_polarizations = \
                np.sort([pol for pol in list_of_polarizations] +
                        [np.bytes_(output_pol)])
            print('output list_of_polarizations:', list_of_polarizations)
            del hdf_obj[key]
            hdf_obj.create_dataset(key, data=list_of_polarizations)

    def remove_pol_recursive(self, hdf_obj: h5py.Group, key: str, pol: str,
                             exclude_string: str = '',
                             product_type: str = ''):

        h5_element = hdf_obj[key]

        if key == pol:
            path = f'{hdf_obj.name}/{pol}'
            del hdf_obj[path]
            return

        if product_type == 'GCOV' and f'{key}{key}' == pol:
            path = f'{hdf_obj.name}/{pol}{pol}'
            del hdf_obj[path]
            return

        if isinstance(h5_element, h5py.Group):

            for sub_key in h5_element.keys():
                if exclude_string and exclude_string in sub_key:
                    continue
                self.remove_pol_recursive(h5_element, sub_key, pol,
                                          exclude_string,
                                          product_type)

        elif pol in POL_LIST and key == 'listOfPolarizations':
            self.print(f'*** H5 path: {h5_element.name}')
            list_of_polarizations = h5_element[()]
            list_of_polarizations_decoded = \
                [p.decode() for p in list_of_polarizations]

            print('input list_of_polarizations:',
                  list_of_polarizations_decoded)

            if pol not in list_of_polarizations_decoded:
                print('*** flag_pol_found. Skipping.')
                return

            list_of_polarizations_decoded.remove(pol)
            print('output list_of_polarizations:',
                  list_of_polarizations_decoded)
            del hdf_obj[key]
            new_list_of_polarizations = \
                [np.bytes_(pol) for pol in list_of_polarizations_decoded]
            hdf_obj.create_dataset(
                key, data=new_list_of_polarizations)


def get_datetime_from_isoformat(ref_epoch):
    ref_epoch = datetime.datetime.strptime(
        ref_epoch.isoformat().split('.')[0],
        "%Y-%m-%dT%H:%M:%S")
    return ref_epoch


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        with PlantIsce3Util(parser, argv) as self_obj:
            ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
