import os
import sys
import time
import datetime
import json
import gc

import plant_isce3
import importlib
from collections.abc import Sequence
from plant_isce3.readers import open_product
from nisar.workflows.geogrid import _grid_size
import numpy as np
import isce3
from osgeo import osr, gdal, gdal_array

from plant_isce3.readers.orbit import load_orbit_from_xml

import plant

DEFAULT_ISCE3_TEMPORARY_FORMAT = 'GTiff'


def add_arguments(parser,
                  abs_cal_factor=0,
                  burst_ids=0,

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

    width_ml = int(input_raster.width // nlooks_x)
    length_ml = int(input_raster.length // nlooks_y)

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

        n_blocks = int(np.ceil(float(input_raster.length) / block_nlines))
    else:
        block_nlines = input_raster.length
        n_blocks = 1

    if verbose:
        print('block number of lines (rounded to nlooks az):', block_nlines)

    for band in range(nbands):

        with plant.PrintProgress(n_blocks) as progress_obj:

            for block in range(n_blocks):
                progress_obj.print_progress(block)

                start_line = block_nlines * block
                end_line = min([block_nlines * (block + 1),
                                input_raster.length + 1])
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
                                              projection=projection)
    if plant_geogrid_obj.has_valid_coordinates():

        if verbose:
            print('geotransform (original):', geotransform)
        geotransform[1] = geotransform[1] * nlooks_x
        geotransform[5] = geotransform[5] * nlooks_y

        if verbose:
            print('geotransform (multilooked):', geotransform)
            print('projection:', projection)

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
            self.sensor_name = 'NISAR'
            self.nisar_product_obj = open_product(
                self.input_file)
            self.frequency = self.get_frequency_str()
            return

        if self.input_file.endswith('.zip'):
            self.sensor_name = 'Sentinel-1'
            self.load_sentinel_1_bursts()
            return

        print(f'ERROR file not recognized: {self.input_file}')

    def get_frequency_str(self):
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

        frequency_str = frequency_list[0]

        return frequency_str

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
                                            radar_grid.ref_epoch())
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
                frequency = self.get_frequency_str()
            return self.nisar_product_obj.getRadarGrid(
                frequency)

        if self.sensor_name == 'Sentinel-1':
            return self.burst.as_isce3_radargrid()

        print(f'ERROR sensor not supported: {self.sensor_name}')

    def get_radar_grid_ml(self, frequency=None):
        radar_grid = self.get_radar_grid()

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

            orbit = load_orbit_from_xml(orbit_file, radar_grid.ref_epoch())

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

        h5_obj = plant.h5py_file_wrapper(self.input_file, *args, **kwargs)
        ret = h5_obj[path][()]
        h5_obj.close()

        return ret


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
        import shapely.wkt

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

    def get_input_raster_from_nisar_slc(self, *args, **kwargs):

        with plant.PlantIndent():
            input_raster = self._get_input_raster_from_nisar_slc(*args,
                                                                 **kwargs)

        return input_raster

    def get_nlooks(self, frequency=None):

        if frequency is None:
            frequency = self.get_frequency_str()

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

    def _get_input_raster_from_nisar_slc(self, input_raster=None,
                                         input_file=None,
                                         plant_product_obj=None):

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

        if input_raster is not None:

            if flag_transform_input_raster is not False:

                flag_apply_transformation = \
                    self.plant_transform_obj.flag_apply_transformation()
                image_obj = self.read_image(input_raster)

            else:
                flag_apply_transformation = False
                image_obj = plant.read_image(input_raster)

            if flag_apply_transformation:
                temp_file = plant.get_temporary_file(append=True,
                                                     ext='vrt')
                self.print(f'*** creating temporary file: {temp_file}')
                self.save_image(image_obj, temp_file, force=True,
                                output_format='VRT')
                input_raster = temp_file

            if flag_symmetrize and symmetrize_bands is None:
                self.print('ERROR symmetrization option with input raster'
                           ' requires the parameter --symmetrize-bands')
                return
            elif flag_symmetrize:
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
                    append=True, suffix='_input_raster_symmerized', ext='tif')
                image_obj = self.read_image(input_raster)

                self._get_symmetrized_input_raster(
                    image_obj, temp_file, temp_symmetrized_file,
                    hv_band=hv_band, vh_band=vh_band,
                    output_format='TIFF')
                input_raster = temp_file

        else:
            frequency_str = plant_product_obj.get_frequency_str()
            self.print(f'selecting product frequency: {frequency_str}')
            nisar_product_obj = open_product(input_file)
            if nisar_product_obj.getProductLevel() == "L1":
                imagery_path = (f'{nisar_product_obj.SwathPath}/'
                                f'frequency{frequency_str}')
            else:
                imagery_path = (f'{nisar_product_obj.GridPath}/'
                                f'frequency{frequency_str}')

            if flag_symmetrize:
                hv_ref = f'HDF5:{input_file}:{imagery_path}/HV'
                vh_ref = f'HDF5:{input_file}:{imagery_path}/VH'
                temp_symmetrized_file = self._symmetrize_cross_pols(
                    hv_ref, vh_ref)

            else:
                temp_symmetrized_file = None

            raster_file = f'NISAR:{input_file}:{frequency_str}'
            temp_file = plant.get_temporary_file(append=True,
                                                 ext='vrt')
            self.print(f'*** creating temporary file: {temp_file}')
            image_obj = self.read_image(raster_file)

            self._get_symmetrized_input_raster(
                image_obj, temp_file, temp_symmetrized_file,
                output_format='VRT')
            input_raster = temp_file

        nlooks_y, nlooks_x = self.get_nlooks(frequency_str)

        if nlooks_y > 1 or nlooks_x > 1:

            self.print('multilooking input file')
            dtype_str = plant.get_dtype_name(image_obj.dtype)
            filter_kwargs = {}

            self.print('data type:', dtype_str)
            if ('COMPLEX' in dtype_str.upper() or
                    'CFLOAT' in dtype_str.upper()):
                exponent = 2

                filter_kwargs['transform_square'] = True
            else:
                exponent = 1

            self.print('exponent:', exponent)
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

    def get_frequency_str(self):

        frequency = self.getattr2('frequency')

        return frequency

    def get_radar_grid_ml(self, radar_grid, frequency=None):

        if self.select_row is not None or self.select_col is not None:
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

    def get_mask_ctable(self, mask_array):
        mask_ctable = gdal.ColorTable()

        mask_ctable.SetColorEntry(0, (175, 175, 175))

        mask_ctable.SetColorEntry(255, (0, 0, 0))

        if not self.cmap:
            self.cmap = 'viridis'

        n_subswaths = np.max(mask_array[(mask_array != 255)])
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
        if verbose:
            print(f'PLAnT (API-completed) - {command_line}'
                  f'{ret_str}')

    plant.plant_config.current_script = current_script

    gc.collect()
    return ret


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
        verbose = kwargs.get('verbose', None) and not (flag_mute is True)
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
