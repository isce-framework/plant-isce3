#!/usr/bin/env python3

import os
import shutil

import numpy as np
import plant
import plant_isce3
import datetime
import h5py
import isce3

from nisar.products.readers import open_product

POL_LIST = ['HH', 'HV', 'VH', 'VV', 'RH', 'RV']


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=1,
                            cmap=1,
                            band=1,
                            default_output_options=1,
                            default_flags=1,
                            output_format=1,
                            multilook=1,
                            output_file=1,
                            output_dir=1)

    plant_isce3.add_arguments(parser,
                              burst_ids=1,
                              orbit_files=1,
                              frequency=1)

    group = parser.add_mutually_exclusive_group(required=True)

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

    group.add_argument('--orbit-kml',
                       dest='orbit_kml_file',
                       action='store_true',
                       help=("Save a KML file containing the product's orbit"
                             " ephemeris"))

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

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        if (not self.input_file.endswith('.h5') and
                not self.input_file.endswith('.nc') and
                not self.input_file.endswith('.SAFE') and
                not self.input_file.endswith('.zip')):

            self.run_raster_as_input()
            plant.append_output_file(self.output_file)
            return self.output_file

        plant_product_obj = self.load_product()
        if self.orbit_kml_file:
            self.save_orbit_kml(plant_product_obj)

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

    def run_nisar_as_input(self, plant_product_obj):
        nisar_product_obj = open_product(self.input_file)

        if self.flag_all_layers or self.flag_all_secondary_layers:
            if self.frequency is not None:

                self.nlooks_az, self.nlooks_rg = \
                    self._get_nlooks(self.frequency)

                suffix = f'_freq_{self.frequency.lower()}'

                return self.save_all_layers(nisar_product_obj,
                                            plant_product_obj,
                                            suffix)

            else:
                frequencies = nisar_product_obj.polarizations.keys()
                for freq in frequencies:
                    self.print(f'## processing frequency {freq}')
                    suffix = f'_freq_{freq.lower()}'
                    self.frequency = freq

                    self.nlooks_az, self.nlooks_rg = self._get_nlooks(freq)

                    self.save_all_layers(nisar_product_obj,
                                         plant_product_obj,
                                         suffix)
                self.frequency = None

            return

        if self.frequency is None:
            freq_pol_dict = nisar_product_obj.polarizations
            self.frequency = list(freq_pol_dict.keys())[0]

        elif self.mask_file:
            self.save_mask(nisar_product_obj)

        elif self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(nisar_product_obj)

        elif self.rtc_gamma_to_sigma_file:

            self.save_nisar_layer('rtcGammaToSigmaFactor', nisar_product_obj)

        elif self.number_of_looks_file:

            self.save_nisar_layer('numberOfLooks', nisar_product_obj)

        elif self.data_file:
            self.save_data()

        elif self.runconfig_file:
            self.save_runconfig_file(nisar_product_obj)

        else:
            if self.input_file != self.output_file:

                input_file_obj = h5py.File(self.input_file, 'r')
                input_file_obj.close()
                shutil.copyfile(self.input_file, self.output_file)

            if self.copy_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.copy_pol[0],
                                            output_pol=self.copy_pol[1])
                    print('done')

            if self.remove_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.remove_pol)
                    print('done')

            if self.move_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.move_pol[0],
                                            output_pol=self.move_pol[1])
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.move_pol[0])
                    print('done')

            print(f'# file saved: {self.output_file}')

    def _get_nlooks(self, freq):
        freq_lower = freq.lower()
        if self.getattr2(f'nlooks_y_{freq_lower}') is not None:
            nlooks_y = self.getattr2(f'nlooks_y_{freq_lower}')
        elif self.getattr2('nlooks_y') is not None:
            nlooks_y = self.getattr2('nlooks_y')
        else:
            nlooks_y = 1

        if self.getattr2(f'nlooks_x_{freq_lower}') is not None:
            nlooks_x = self.getattr2(f'nlooks_x_{freq_lower}')
        elif self.getattr2('nlooks_x') is not None:
            nlooks_x = self.getattr2('nlooks_x')
        else:
            nlooks_x = 1

        return nlooks_y, nlooks_x

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

        self.output_file = os.path.join(self.output_dir,
                                        f'{prefix}mask{suffix}.{ext}')
        self.save_mask(nisar_product_obj)

        if nisar_product_obj.productType == 'GCOV':
            self.output_file = os.path.join(
                self.output_dir,
                f'{prefix}rtcGammaToSigma{suffix}.{ext}')
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
            self.save_data(plant_product_obj)

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

    def save_data(self, plant_product_obj):

        input_raster = self.get_input_raster_from_nisar_slc(
            input_raster=None,
            plant_product_obj=plant_product_obj)

        self.save_image(input_raster, output_file=self.output_file,
                        force=True)

        plant.append_output_file(self.output_file)

    def save_mask(self, nisar_product_obj=None,
                  image_obj=None):

        image_ref = self.get_grids_ref(
            'mask', nisar_product_obj, image_obj,
            valid_products=['GCOV', 'GSLC'])

        image_obj = self.read_image(image_ref)

        mask_array_obj = plant.filter_data(image_obj,
                                           nlooks=[self.nlooks_az,
                                                   self.nlooks_rg])
        del image_obj

        mask_array_obj.image = np.asarray(mask_array_obj.image,
                                          dtype=np.uint8)
        mask_array_obj.dtype = np.uint8

        mask_ctable = self.get_mask_ctable(mask_array_obj.image)

        self.save_image(mask_array_obj, output_file=self.output_file,
                        out_null=255, ctable=mask_ctable)
        plant.append_output_file(self.output_file)

    def get_grids_ref(self, layer_name, nisar_product_obj, image_obj,
                      valid_products=['GCOV', 'GSLC']):
        if image_obj is not None:
            return image_obj
        if nisar_product_obj.productType not in valid_products:
            error_msg = (f'ERROR cannot save layer "{layer_name}" for'
                         ' product type'
                         f' "{nisar_product_obj.productType}".')
            print(error_msg)
            raise ValueError(error_msg)

        grid_path = (f'{nisar_product_obj.GridPath}'
                     f'/frequency{self.frequency}/{layer_name}')
        image_ref = f'NETCDF:{self.input_file}:{grid_path}'

        return image_ref

    def save_nisar_layer(self, layer_name, nisar_product_obj=None,

                         image_obj=None):

        image_ref = self.get_grids_ref(
            layer_name, nisar_product_obj, image_obj)

        if self.nlooks_az != 1 or self.nlooks_rg != 1:

            temp_file = plant.get_temporary_file(ext='.tif', append=True)
            plant_isce3.multilook_isce3(image_ref,
                                        output_file=temp_file,
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
            'layoverShadowMask', nisar_product_obj, image_obj)

        layover_shadow_mask_ctable = self.get_layover_shadow_mask_ctable()

        self.save_image(image_obj, output_file=self.output_file,
                        out_null=255, ctable=layover_shadow_mask_ctable)
        plant.append_output_file(self.output_file)

    def save_lut(self, h5_path, pol_list=[], flag_skip_if_error=True):
        if '{pol}' in h5_path:
            for pol in pol_list:
                output_file_orig = self.output_file
                self.output_file = self.output_file.replace('{pol}', pol)
                self.save_lut(h5_path.replace('{pol}', pol))
                self.output_file = output_file_orig
                return
        ref = f'NETCDF:{self.input_file}:{h5_path}'
        try:
            image_obj = plant.read_image(ref)
            self.save_image(image_obj, output_file=self.output_file)
            plant.append_output_file(self.output_file)
        except BaseException:
            if flag_skip_if_error:
                return
            self.print(f'ERROR fail to open dataset "{h5_path}" from '
                       f' "{self.input_file}"')

    def save_runconfig_file(self, nisar_product_obj):

        h5_obj = h5py.File(self.input_file, 'r')

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

            h5_obj = h5py.File(self.input_file, 'r')

            flag_has_polygon = True
            polygon_dataset = '//science/LSAR/identification/boundingPolygon'
            polygon_str = str(h5_obj[polygon_dataset][()].decode('utf-8'))
            h5_obj.close()
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

    def copy_pol_recursive(self, hdf_obj: h5py.Group, key: str, input_pol: str,
                           output_pol: str):

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
                self.copy_pol_recursive(h5_element, sub_key, input_pol,
                                        output_pol)

        elif input_pol in POL_LIST and key == 'listOfPolarizations':
            list_of_polarizations = h5_element[()]
            print('input list_of_polarizations:', list_of_polarizations)
            flag_input_pol_found = any([pol.decode() == input_pol
                                        for pol in list_of_polarizations])

            if not flag_input_pol_found:
                print(f'WARNING input pol {input_pol} not found in the list'
                      f'of polarization at: "{h5_element.name}"')
                return
            flag_output_pol_found = any([pol.decode() == output_pol
                                        for pol in list_of_polarizations])

            if flag_output_pol_found:
                print(f'WARNING output pol {output_pol} already exists in list'
                      f'of polarization at: "{h5_element.name}"')
                return
            list_of_polarizations = np.sort(np.append(list_of_polarizations,
                                            np.bytes_(output_pol)))
            print('output list_of_polarizations:', list_of_polarizations)
            del hdf_obj[key]
            hdf_obj.create_dataset(key, data=list_of_polarizations)

    def remove_pol_recursive(self, hdf_obj: h5py.Group, key: str, pol: str):

        h5_element = hdf_obj[key]

        if key == pol:
            path = f'{hdf_obj.name}/{pol}'
            del hdf_obj[path]
            return

        if isinstance(h5_element, h5py.Group):

            for sub_key in h5_element.keys():
                self.remove_pol_recursive(h5_element, sub_key, pol)

        elif pol in POL_LIST and key == 'listOfPolarizations':
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
            hdf_obj.create_dataset(
                key, data=np.bytes_(list_of_polarizations_decoded))


def get_datetime_from_isoformat(ref_epoch):
    ref_epoch = datetime.datetime.strptime(
        ref_epoch.isoformat().split('.')[0],
        "%Y-%m-%dT%H:%M:%S")
    return ref_epoch


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Util(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
