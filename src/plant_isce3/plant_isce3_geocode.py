#!/usr/bin/env python3

import os
import plant
from osgeo import gdal
import numpy as np
import isce3
from nisar.products.readers import SLC

PSP_NULL = 0


def get_parser():
    '''
    Command line parser.
    '''
    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,  # 2 if required
                            dem_file=2,
                            default_options=1,
                            geo=1,
                            multilook=1,
                            output_file=2)

    parser.add_argument('--input-raster',
                        dest='input_raster',
                        type=str,
                        help='Input raster.')

    parser_transform_input_raster = parser.add_mutually_exclusive_group()
    parser_transform_input_raster.add_argument(
        '--transform-input-raster',
        action='store_true',
        default=None,
        dest='flag_transform_input_raster',
        help='Let PLAnT requested transformations (e.g. crop, absolute,'
        ' phase) to input raster (default)')

    parser_transform_input_raster.add_argument(
        '--no-transform-input-raster',
        '--do-not-transform-input-raster',
        action='store_false',
        dest='flag_transform_input_raster',
        help='Prevent PLAnT from applying requested transformations'
        ' (e.g. crop, absolute, phase) to input raster')

    parser.add_argument('--exponent',
                        dest='exponent',
                        type=int,
                        help='Exponent for geocoding.')

    parser.add_argument('--input-rtc',
                        dest='input_rtc',
                        type=str,
                        help='Input RTC area factor.')

    parser.add_argument('--epsg',
                        dest='epsg',
                        # action='store',
                        type=int,
                        help='EPSG code for output grids.')

    parser.add_argument('--offset-az',
                        dest='offset_az',
                        type=str,
                        help='Azimuth offset.')

    parser.add_argument('--offset-rg',
                        dest='offset_rg',
                        type=str,
                        help='Range offset.')

    parser.add_argument('--cov',
                        '--cov-matrix',
                        '--covariance-matrix',
                        dest='covariance_matrix',
                        type=str,
                        help='Output covariance matrix [C] directory.')

    parser.add_argument('--abs-cal-factor',
                        '--abs-calibration-factor',
                        '--calibration-factor',
                        dest='abs_cal_factor',
                        type=float,
                        help='Absolute calibration factor')

    parser.add_argument('--symmetrize',
                        action='store_true',
                        dest='flag_symmetrize',
                        help='Apply polarimetric symmetrization')

    parser.add_argument('--symmetrize-bands',
                        dest='symmetrize_bands',
                        nargs=2,
                        type=int,
                        help='Bands (starting from 0) to symmetrize before geocoding.')

    parser.add_argument('--list-of-polarizations',
                        dest='list_of_polarizations',
                        nargs='+',
                        type=str,
                        help='List of input polarizations (only'
                        ' applicable for covariance matrix).')

    parser.add_argument('--gslc',
                        action='store_true',
                        dest='flag_gslc',
                        help='Generated geocoded SLC')

    parser.add_argument('--stats',
                        '--compute-stats',
                        action='store_true',
                        dest='flag_compute_stats',
                        help='Compute stats (min, max, mean, and stddev)')

    parser_az_baseband_doppler = parser.add_mutually_exclusive_group()
    parser_az_baseband_doppler.add_argument(
        '--az_baseband_doppler',
        action='store_true',
        default=None,
        dest='flag_az_baseband_doppler',
        help='Baseband Doppler before interpolation/averaging')

    parser_az_baseband_doppler.add_argument(
        '--no-az_baseband_doppler',
        action='store_false',
        dest='flag_az_baseband_doppler',
        help='Do not baseband Doppler before interpolation/'
             ' averaging')

    parser_flatten = parser.add_mutually_exclusive_group()
    parser_flatten.add_argument('--flatten',
                                action='store_true',
                                default=None,
                                dest='flatten',
                                help='Apply phase flattening')

    parser_flatten.add_argument('--no-flatten',
                                action='store_false',
                                dest='flatten',
                                help='Do not apply phase flattening')

    parser_output_mode = parser.add_mutually_exclusive_group()
    parser_output_mode.add_argument('--point',
                                    '--interp',
                                    action='store_true',
                                    dest='output_mode_interp',
                                    help='Use interp mode')

    parser_output_mode.add_argument('--point-gamma-naught',
                                    '--interp-gamma-naught',
                                    action='store_true',
                                    dest='output_mode_interp_gamma_naught',
                                    help='Use interp mode and apply'
                                    ' radiometric terrain correction')

    parser_output_mode.add_argument('--area',
                                    '--area-proj',
                                    '--area-projection',
                                    action='store_true',
                                    dest='output_mode_area',
                                    help='Use area mode')

    parser_output_mode.add_argument('--area-proj-gamma-naught',
                                    '--area-projection-gamma-naught',
                                    '--area-gamma-naught',
                                    action='store_true',
                                    dest='output_mode_area_gamma_naught',
                                    help='Use area mode and apply radiometric'
                                    ' terrain correction')

    parser.add_argument('--interp-method',
                        dest='interp_method',
                        type=str,
                        help='Interpolation method. Options:'
                        ' sinc, bilinear, bicubic, nearest, biquintic')

    parser.add_argument('--terrain', '--terrain-type',
                        '--rtc',
                        dest='terrain_correction_type',
                        type=str,
                        help="type of radiometric terrain correction: "
                        # "'beta_naught', "
                        # "'sigma-naught'"
                        # "'sigma-naught-norlim', "
                        # "'sigma-naught-ahmed', "
                        # "'sigma-naught-ulander', "i
                        "'gamma-naught-david-small', "
                        # "'gamma-naught-norlim', "
                        # "'gamma-naught', "
                        "'gamma-naught-area-projection' "
                        # "'gamma-naught-ulander' "
                        "(default: %(default)s)",
                        default='gamma-naught-area-projection')

    parser.add_argument('--native-doppler',
                        dest='native_doppler',
                        default=False,
                        action='store_true',
                        help='Native Doppler.')

    parser.add_argument('--double-radar-grid-sampling',
                        '--upsample-radar-grid',
                        dest='flag_upsample_radar_grid',
                        default=None,
                        action='store_true',
                        help='Double radar grid sampling.')

    '''
    parser.add_argument('--full-covariance',
                        dest='full_covariance',
                        action='store_true',
                        help='Include off-diagonal terms to the covariance matrix.')
    '''

    parser.add_argument('--out-off-diag-terms',
                        '--out-off-diagonal-terms',
                        dest='out_off_diag_terms',
                        type=str,
                        help='Output off-diagonal terms.')

    parser.add_argument('--upsampling',
                        dest='geogrid_upsampling',
                        type=float,
                        help='Geogrid upsample factor.')

    parser.add_argument('--rtc-upsampling',
                        dest='rtc_upsampling',
                        type=float,
                        help='RTC geogrid upsample factor.')

    parser.add_argument('--rtc-min-value-db',
                        dest='rtc_min_value_db',
                        type=float,
                        help='DEM upsample factor.')

    parser.add_argument('--input-radiometry',
                        '--input-terrain-radiometry',
                        dest='input_terrain_radiometry',
                        type=str,
                        help='Input data radiometry. Options:'
                        'beta or sigma-ellipsoid')

    parser.add_argument('--output-radiometry',
                        '--output-terrain-radiometry',
                        dest='output_terrain_radiometry',
                        type=str,
                        help='Output data radiometry. Options:'
                        'sigma-naught or gamma-naught')

    parser.add_argument('--out-geo-vertices',
                        dest='out_geo_vertices',
                        type=str,
                        help='Output geo vertices file')

    parser.add_argument('--out-dem-vertices',
                        dest='out_dem_vertices',
                        type=str,
                        help='Output interpolated DEM file')

    parser.add_argument('--out-geo-nlooks',
                        dest='out_geo_nlooks',
                        type=str,
                        help='Output geo nlooks file')

    parser.add_argument('--out-geo-rtc',
                        dest='out_geo_rtc',
                        type=str,
                        help='Output geo RTC file')

    '''
    parser.add_argument('--out-geo-local-inc-angle',
                        '--out-geo-local-incidence-angle',
                        '--out-geo-lia',
                        dest='out_geo_local_inc_angle',
                        type=str,
                        help='Output geo local-incidence angle file')
    '''

    parser.add_argument('--out-geo-rtc-anf-to-sigma0',
                        '--out-geo-rtc-anf-sigma0',
                        '--out-geo-rtc-sigma0',
                        dest='out_geo_rtc_anf_to_sigma0',
                        type=str,
                        help='Output geo RTC ANF to sigma0 file')

    parser.add_argument('--out-rtc',
                        '--output-rtc',
                        dest='output_rtc',
                        type=str,
                        help='Output RTC ANF file (in slant-range)')

    parser.add_argument('--memory-mode',
                        dest='memory_mode',
                        type=str,
                        choices=['auto',
                                 'single-block',
                                 'blocks-geogrid',
                                 'blocks-geogrid-and-radargrid'],
                        help='Memory mode')

    parser.add_argument('--min-block-size',
                        type=int,
                        dest='min_block_size',
                        help='Minimum block size in Bytes')

    parser.add_argument('--max-block-size',
                        type=int,
                        dest='max_block_size',
                        help='Maximum block size in Bytes')

    parser.add_argument('--clip-min',
                        type=float,
                        dest='clip_min',
                        help='Clip (limit) min output values')

    parser.add_argument('--clip-max',
                        type=float,
                        dest='clip_max',
                        help='Clip (limit) max output values')

    parser.add_argument('--nlooks-min',
                        '--min-nlooks',
                        type=float,
                        dest='min_nlooks',
                        help='Minimum number of looks. Geogrid data'
                        ' below this limit will be set to NaN.')

    parser.add_argument('--geo2rdr-threshold',
                        type=float,
                        # default=1e-8,
                        dest='geo2rdr_threshold',
                        help='Range convergence threshold for geo2rdr')

    parser.add_argument('--geo2rdr-num-iter',
                        '--geo2rdr-numiter',
                        type=float,
                        # default=100,
                        dest='geo2rdr_num_iter',
                        help='Maximum number of iterations for geo2rdr')

    '''
    in_amplitude_group = parser.add_mutually_exclusive_group(required=False)
    in_amplitude_group.add_argument('--in-intensity', '--in-int',
                                    dest='in_amplitude',
                                    default=False,
                                    action='store_true',
                                    help='Input image is intensity')
    in_amplitude_group.add_argument('--in-amplitude', '--in-amp',
                                    dest='in_amplitude',
                                    action='store_true',
                                    help='Input image is amplitude')
    '''

    return parser


class PlantIsce3Geocode(plant.PlantScript):

    def __init__(self, parser, argv=None):
        '''
        class initialization
        '''
        super().__init__(parser, argv)

    def _get_raster(self, filename, nbands=1, gdal_dtype=gdal.GDT_Float32,
                    width=None, length=None):
        if not filename:
            return
        print(f'creating file: {filename}')
        if width is None:
            width = self.lon_size  # *self.geogrid_upsampling
        if length is None:
            length = self.lat_size  # *self.geogrid_upsampling
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # output_format = plant.get_output_format(filename)
        # print('output format:', output_format)
        output_format = "ENVI"

        output_obj = isce3.io.Raster(
            filename,
            int(width),
            int(length),
            int(nbands),
            int(gdal_dtype),
            output_format)
        if not plant.isfile(filename):
            self.print(f'ERROR creating {filename}')
            return

        return output_obj

    def run(self):
        '''
        run main method
        '''

        if self.input_key and self.input_key == 'B':
            frequency_str = 'B'
        else:
            frequency_str = 'A'

        ret_dict = self._get_input_raster_from_nisar_slc(
            self.input_raster)
        input_raster = ret_dict['input_raster']
        input_raster_obj = isce3.io.Raster(input_raster)
        ellipsoid = isce3.core.Ellipsoid()

        if self.nlooks_az is None:
            self.nlooks_az = 1
        if self.nlooks_rg is None:
            self.nlooks_rg = 1

        slc_obj = SLC(hdf5file=self.input_file)
        orbit = slc_obj.getOrbit()
        doppler = self._get_doppler(slc_obj)

        # Get radar grid
        radar_grid = self._get_radar_grid(slc_obj, frequency_str)

        # init Geocode object according to raster type
        if input_raster_obj.datatype() == gdal.GDT_Float32:
            geo = isce3.geocode.GeocodeFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_Float64:
            geo = isce3.geocode.GeocodeFloat64()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
            geo = isce3.geocode.GeocodeCFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
            geo = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            # error_channel.log(err_str)
            raise NotImplementedError(err_str)

        # init geocode members
        geo.orbit = orbit
        geo.ellipsoid = ellipsoid

        dem_raster = isce3.io.Raster(self.dem_file)
        if self.epsg is None:
            if dem_raster.get_epsg() == 0 or dem_raster.get_epsg() < -9000:
                dem_raster.set_epsg(4326)
            self.epsg = dem_raster.get_epsg()

        proj_win_from_dem = None
        if self.lon_arr is None or self.lat_arr is None:
            proj_win_from_dem = isce3.py_computeDEMBounds(
                orbit, ellipsoid, doppler,
                radar_grid, 0, 0, radar_grid.width, radar_grid.length,
                margin=np.radians(0.01))
            proj_win_from_dem = np.degrees(proj_win_from_dem)
        self._update_geogrid(proj_win_from_dem)

        lat_size = self.lat_size if plant.isvalid(self.lat_size) else -9999
        lon_size = self.lon_size if plant.isvalid(self.lon_size) else -9999
        y0_orig = self.lat_arr[1]
        # if plant.isvalid(y0_orig) and plant.isvalid(self.step_lat):
        #     y0_orig = y0_orig + abs(self.step_lat)/2.0
        x0_orig = self.lon_arr[0]
        # if plant.isvalid(x0_orig) and plant.isvalid(self.step_lon):
        #     x0_orig = x0_orig - self.step_lon/2.0

        if self.epsg == 4326 and not plant.isvalid(self.step_lon):
            self.step_lon = plant.m_to_deg_lon(30.)
        if self.epsg == 4326 and not plant.isvalid(self.step_lat):
            self.step_lat = plant.m_to_deg_lat(30.)

        print(f'x0_orig: {x0_orig}')
        print(f'y0_orig: {y0_orig}')
        print(f'self.step_lon: {self.step_lon}')
        print(f'self.step_lat: {self.step_lat}')
        print(f'lon_size: {lon_size}')
        print(f'lat_size: {lat_size}')
        print(f'epsg: {self.epsg}')
        geo.geogrid(x0_orig,
                    y0_orig,
                    self.step_lon,
                    -abs(self.step_lat),
                    int(lon_size),
                    int(lat_size),
                    self.epsg)

        geo.update_geogrid(radar_grid, dem_raster)

        x0 = geo.geogrid_start_x  # +geo.geogrid_spacing_x/2.0
        y0 = geo.geogrid_start_y  # +geo.geogrid_spacing_y/2.0
        dx = geo.geogrid_spacing_x
        dy = geo.geogrid_spacing_y
        geogrid_width = abs(geo.geogrid_width)
        geogrid_length = abs(geo.geogrid_length)
        self.lon_arr = [x0, x0 + dx * geogrid_width]
        self.lon_size = geogrid_width
        self.lat_arr = [y0 + dy * geogrid_length, y0]
        self.lat_size = geogrid_length
        self.step_lat = -dy
        self.step_lon = dx

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        input_dtype = input_raster_obj.datatype()

        # GSLC
        if (self.flag_gslc and self.exponent is not None and
                self.exponent != 1):
            self.print('ERROR GSLC accepts only unitary exponent')
            return
        elif self.flag_gslc:
            self.exponent = 1
            output_dtype = input_dtype
            # native_doppler = slc_obj.getDopplerCentroid()
            # product_type = 'GSLC'
            if self.interp_method is None:
                self.interp_method = 'sinc'
            if self.flag_az_baseband_doppler is None:
                self.flag_az_baseband_doppler = True
                if self.flatten is None:
                    self.flatten = True

        # GCOV
        else:
            # exponent = 2
            # output_dtype = gdal.GDT_Float32
            # native_doppler = isce3.core.LUT2d()
            # product_type = 'GCOV'
            if self.interp_method is None:
                self.interp_method = 'biquintic'

            if self.exponent is None:
                self.exponent = 2 if self.transform_square else 0

            if self.exponent % 2 == 0 and input_dtype == gdal.GDT_CFloat32:
                output_dtype = gdal.GDT_Float32
                self.transform_square = True
            elif self.exponent % 2 == 0 and input_dtype == gdal.GDT_CFloat64:
                output_dtype = gdal.GDT_Float64
                self.transform_square = True
            else:
                output_dtype = input_dtype

        # geo2rdr range threshold
        if self.geo2rdr_threshold is None:
            self.geo2rdr_threshold = 1e-08
        print('geo2rdr threshold:', self.geo2rdr_threshold)

        # geo2rdr Newton Raphson number of iteration
        if self.geo2rdr_num_iter is None:
            self.geo2rdr_num_iter = 100
        print('geo2rdr num iter:', self.geo2rdr_num_iter)

        # set geocoding parameters
        geo.doppler = doppler
        # geo.native_doppler = native_doppler
        geo.threshold_geo2rdr = self.geo2rdr_threshold
        geo.numiter_geo2rdr = self.geo2rdr_num_iter
        # geo.num_iter_geo2rdr = self.geo2rdr_num_iter

        print('interp method:', self.interp_method)
        if (self.interp_method is not None and
                self.interp_method.upper() == 'SINC'):
            interp_method = isce3.core.DataInterpMethod.SINC
        elif (self.interp_method is not None and
                self.interp_method.upper() == 'BILINEAR'):
            interp_method = isce3.core.DataInterpMethod.BILINEAR
        elif (self.interp_method is not None and
                self.interp_method.upper() == 'BICUBIC'):
            interp_method = isce3.core.DataInterpMethod.BICUBIC
        elif (self.interp_method is not None and
                self.interp_method.upper() == 'NEAREST'):
            interp_method = isce3.core.DataInterpMethod.NEAREST
        else:
            interp_method = isce3.core.DataInterpMethod.BIQUINTIC

        self.print(f'*** exponent: {self.exponent}')
        self.print(f'*** input_dtype: {input_dtype}')
        self.print(f'*** output_dtype: {output_dtype}')
        if output_dtype == gdal.GDT_Float32:
            print('*** output dtype: float32')
        elif output_dtype == gdal.GDT_Float64:
            print('*** output dtype: float64')
        elif output_dtype == gdal.GDT_CFloat32:
            print('*** output dtype: cfloat32')
        elif output_dtype == gdal.GDT_CFloat64:
            print('*** output dtype: cfloat64')

        nbands = input_raster_obj.num_bands
        self.print(f'*** nbands: {nbands}')

        if self.geogrid_upsampling is None:
            self.geogrid_upsampling = 1

        print('self.lon_arr:', self.lon_arr)
        print('self.lat_arr:', self.lat_arr)
        print('self.lon_size:', self.lon_size)
        print('self.lat_size:', self.lat_size)
        print('self.step_lon:', self.step_lon)
        print('self.step_lat:', self.step_lat)

        output_raster_obj = isce3.io.Raster(
            self.output_file,
            # gdal.GA_Update,
            self.lon_size,
            self.lat_size,
            nbands,
            output_dtype,
            "ENVI")

        # self.out_off_diag_terms = None
        out_off_diag_terms_obj = None
        if self.out_off_diag_terms or self.covariance_matrix:
            # self.out_off_diag_terms = self.output_file + '_off_diag.tif'

            if not self.out_off_diag_terms:
                self.out_off_diag_terms = plant.get_temporary_file(append=True,
                                                                   ext='bin')

            nbands_off_diag_terms = int((nbands**2-nbands)/2)
            print('nbands_off_diag_terms: ', nbands_off_diag_terms)
            if nbands_off_diag_terms > 0:
                out_off_diag_terms_obj = self._get_raster(
                    self.out_off_diag_terms,
                    gdal_dtype=gdal.GDT_CFloat32,
                    nbands=nbands_off_diag_terms)

        print('*** out geo rtc:', self.out_geo_rtc)
        out_geo_nlooks_obj = self._get_raster(self.out_geo_nlooks)
        out_geo_rtc_obj = self._get_raster(self.out_geo_rtc)
        # out_geo_local_inc_angle_obj = self._get_raster(
        # self.out_geo_local_inc_angle)
        out_geo_rtc_anf_to_sigma0_obj = self._get_raster(
            self.out_geo_rtc_anf_to_sigma0)

        output_rtc_obj = self._get_raster(self.output_rtc,
                                          length=radar_grid.length,
                                          width=radar_grid.width)

        print(f'*** input_raster.width: {input_raster_obj.width}')
        print(f'*** input_raster.length: {input_raster_obj.length}')

        print(f'*** output_raster_obj.width: {output_raster_obj.width}')
        print(f'*** output_raster_obj.length: {output_raster_obj.length}')

        kwargs = {}

        if self.memory_mode == 'single-block':
            kwargs['memory_mode'] = isce3.core.GeocodeMemoryMode.SingleBlock
        elif self.memory_mode == 'blocks-geogrid':
            kwargs['memory_mode'] = isce3.core.GeocodeMemoryMode.BlocksGeogrid
        elif self.memory_mode == 'blocks-geogrid-and-radargrid':
            kwargs['memory_mode'] = \
                isce3.core.GeocodeMemoryMode.BlocksGeogridAndRadarGrid

        if self.rtc_upsampling is not None:
            kwargs['rtc_upsampling'] = self.rtc_upsampling

        if self.min_block_size is not None:
            kwargs['min_block_size'] = self.min_block_size

        if self.max_block_size is not None:
            kwargs['max_block_size'] = self.max_block_size

        if self.min_nlooks:
            kwargs['min_nlooks'] = self.min_nlooks

        if self.offset_az:
            print(f'input RTC: {self.offset_az}')
            offset_az_obj = isce3.io.Raster(self.offset_az)
            kwargs['offset_az'] = offset_az_obj
        else:
            offset_az_obj = None

        if self.offset_rg:
            print(f'input RTC: {self.offset_rg}')
            offset_rg_obj = isce3.io.Raster(self.offset_rg)
            kwargs['offset_rg'] = offset_rg_obj
        else:
            offset_rg_obj = None

        flag_rtc_bilinear_distribution = (
            self.terrain_correction_type is not None and
            (('DAVID' in self.terrain_correction_type.upper() and
             'SMALL' in self.terrain_correction_type.upper()) or
             ('BILINEAR' in self.terrain_correction_type.upper() and
             'DISTR' in self.terrain_correction_type.upper())))

        flag_rtc_area_proj = (
            (self.terrain_correction_type is not None and
             'AREA' in self.terrain_correction_type.upper() and
             'PROJ' in self.terrain_correction_type.upper()) or
            self.output_mode_area_gamma_naught or
            self.output_mode_interp_gamma_naught)

        if flag_rtc_bilinear_distribution:
            kwargs['rtc_algorithm'] = \
                isce3.geometry.RtcAlgorithm.RTC_BILINEAR_DISTRIBUTION
        elif flag_rtc_area_proj:
            kwargs['rtc_algorithm'] = \
                isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION
        else:
            kwargs['rtc_algorithm'] = \
                isce3.geometry.RtcAlgorithm.RTC_AREA_PROJECTION

        flag_output_terrain_radimetry_is_sigma = \
            (self.output_terrain_radiometry is not None and
             'sigma' in self.output_terrain_radiometry.lower())

        if (out_geo_rtc_anf_to_sigma0_obj and
                flag_output_terrain_radimetry_is_sigma):
            self.print('ERROR RTC ANF from gamma0 to sigma0 is not implemented'
                       ' for output terrain radiometry as sigma0')
            return

        if flag_output_terrain_radimetry_is_sigma:
            kwargs['output_terrain_radiometry'] = \
                isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
        else:
            kwargs['output_terrain_radiometry'] = \
                isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT

        flag_input_terrain_radiometry_is_sigma = \
            (self.input_terrain_radiometry is not None and
             'sigma' in self.input_terrain_radiometry)

        if flag_input_terrain_radiometry_is_sigma:
            kwargs['input_terrain_radiometry'] = \
                isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            kwargs['input_terrain_radiometry'] = \
                isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

        if self.rtc_min_value_db is not None:
            kwargs['rtc_min_value_db'] = self.rtc_min_value_db

        kwargs['radargrid_nlooks'] = self.nlooks_az * self.nlooks_rg

        if self.abs_cal_factor is not None:
            kwargs['abs_cal_factor'] = self.abs_cal_factor

        if self.clip_min is not None:
            kwargs['clip_min'] = self.clip_min

        if self.clip_max is not None:
            kwargs['clip_max'] = self.clip_max

        if self.flag_az_baseband_doppler is not None:
            kwargs['flag_az_baseband_doppler'] = self.flag_az_baseband_doppler

        if self.flatten is not None:
            kwargs['flatten'] = self.flatten

        if self.flag_upsample_radar_grid is not None:
            kwargs['flag_upsample_radar_grid'] = self.flag_upsample_radar_grid

        if out_off_diag_terms_obj is not None:
            kwargs['out_off_diag_terms'] = out_off_diag_terms_obj

        if self.input_rtc:
            print(f'input RTC: {self.input_rtc}')
            input_rtc_obj = isce3.io.Raster(self.input_rtc)
        else:
            input_rtc_obj = None

        flag_geocode_interp = (self.output_mode_interp or
                               self.output_mode_interp_gamma_naught)
        # print('flag_geocode_interp:', flag_geocode_interp)
        flag_geocode_area_proj = (self.output_mode_area or
                                  self.output_mode_area_gamma_naught)
        # print('flag_geocode_area_proj:', flag_geocode_area_proj)

        if flag_geocode_area_proj or not flag_geocode_interp:
            kwargs['output_mode'] = \
                isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
        else:
            kwargs['output_mode'] = \
                isce3.geocode.GeocodeOutputMode.INTERP

        kwargs['flag_apply_rtc'] = (flag_rtc_bilinear_distribution or
                                    flag_rtc_area_proj)

        if self.flag_compute_stats:
            kwargs['compute_stats'] = self.flag_compute_stats

        if flag_geocode_interp:
            geo_raster_rdr_dem_width = self.lon_size
            geo_raster_rdr_dem_length = self.lat_size
        else:
            geo_raster_rdr_dem_width = (self.lon_size *
                                        self.geogrid_upsampling + 1)
            geo_raster_rdr_dem_length = (self.lat_size *
                                         self.geogrid_upsampling + 1)

        out_geo_vertices_obj = self._get_raster(
            self.out_geo_vertices,
            width=geo_raster_rdr_dem_width,
            length=geo_raster_rdr_dem_length,
            nbands=2)
        out_dem_vertices_obj = self._get_raster(
            self.out_dem_vertices,
            width=geo_raster_rdr_dem_width,
            length=geo_raster_rdr_dem_length,
            nbands=1)

        '''
        if out_geo_local_inc_angle_obj:
            kwargs['out_geo_local_inc_angle'] = out_geo_local_inc_angle_obj,

        if out_geo_rtc_anf_to_sigma0_obj:
            kwargs['out_geo_rtc_anf_to_sigma0'] =
                out_geo_rtc_anf_to_sigma0_obj,
        '''

        # print('geocode kwargs:', kwargs)
        geo.geocode(radar_grid,
                    input_raster_obj,
                    output_raster_obj,
                    dem_raster,
                    geogrid_upsampling=self.geogrid_upsampling,
                    exponent=self.exponent,
                    out_geo_rdr=out_geo_vertices_obj,
                    out_geo_dem=out_dem_vertices_obj,
                    out_geo_nlooks=out_geo_nlooks_obj,
                    out_geo_rtc=out_geo_rtc_obj,
                    out_geo_rtc_gamma0_to_sigma0=out_geo_rtc_anf_to_sigma0_obj,
                    input_rtc=input_rtc_obj,
                    output_rtc=output_rtc_obj,
                    dem_interp_method=interp_method,
                    **kwargs)

        # del input_raster_obj
        del geo
        del output_raster_obj
        ret_dict = {}

        if self.out_off_diag_terms:
            try:
                out_off_diag_terms_obj.close_dataset()
            except:
                pass
            del out_off_diag_terms_obj
            plant.append_output_file(self.out_off_diag_terms)
            ret_dict['out_off_diag_terms'] = self.out_off_diag_terms
        if self.out_geo_vertices:
            del out_geo_vertices_obj
            plant.append_output_file(self.out_geo_vertices)
            ret_dict['out_geo_vertices'] = self.out_geo_vertices
        if self.out_dem_vertices:
            del out_dem_vertices_obj
            plant.append_output_file(self.out_dem_vertices)
            ret_dict['out_dem_vertices'] = self.out_dem_vertices
        if self.out_geo_nlooks:
            del out_geo_nlooks_obj
            plant.append_output_file(self.out_geo_nlooks)
            ret_dict['out_geo_nlooks'] = self.out_geo_nlooks
        if self.out_geo_rtc:
            del out_geo_rtc_obj
            plant.append_output_file(self.out_geo_rtc)
            ret_dict['out_geo_rtc'] = self.out_geo_rtc
        '''
        if self.out_geo_local_inc_angle:
            del out_geo_local_inc_angle_obj
            plant.append_output_file(self.out_geo_local_inc_angle)
            ret_dict['out_geo_local_inc_angle'] = self.out_geo_local_inc_angle
        '''
        if self.output_rtc:
            del output_rtc_obj
            plant.append_output_file(self.output_rtc)
            ret_dict['output_rtc'] = self.output_rtc

        ret_dict['output_file'] = self.output_file
        plant.append_output_file(self.output_file)

        if self.covariance_matrix:
            self._generate_cov_matrix(frequency_str)

        for output_file in ret_dict.values():
            # print('output file:', output_file)
            expected_output_format = plant.get_output_format(output_file)
            image_obj = plant.read_image(output_file)
            actual_output_format = image_obj.file_format
            if expected_output_format != actual_output_format:
                plant.util(output_file, output_file=output_file,
                           output_format=expected_output_format,
                           force=True)
                if actual_output_format != 'ENVI':
                    continue
                envi_header = plant.get_envi_header(output_file)
                if os.path.isfile(envi_header):
                    os.remove(envi_header)

        return ret_dict

    def _generate_cov_matrix(self, frequency_str):
        image_obj = plant.read_image(self.output_file)
        nbands = image_obj.nbands
        width = image_obj.width
        length = image_obj.length

        # gdal_ds = gdal.Open(self.out_off_diag_terms)
        image_off_diag_obj = plant.read_image(self.out_off_diag_terms)
        band_index = 0

        # get list of polarizations
        if self.list_of_polarizations is not None:
            list_of_polarizations = self.list_of_polarizations
        else:
            if self.input_raster:
                input_raster_obj = plant.read_image(self.input_raster)
                list_of_polarizations = []
                for b in range(input_raster_obj.nbands):
                    band_name = input_raster_obj.get_band(band=b).name
                    print('band name: ', band_name)
                    list_of_polarizations.append(band_name)
            else:
                pol_key = ('//science/LSAR/RSLC/swaths/'
                           f'frequency{frequency_str}/listOfPolarizations')
                flag_error = False
                try:
                    list_of_polarizations = plant.read_image(
                        f'HDF5:{self.input_file}:{pol_key}').image[0]
                except:
                    flag_error = True
                if flag_error:
                    pol_key = (f'//science/LSAR/SLC/swaths/'
                               'frequency{frequency_str}/listOfPolarizations')
                    list_of_polarizations = plant.read_image(
                        f'HDF5:{self.input_file}:{pol_key}').image[0]

            try:
                list_of_polarizations = [p.upper()
                                         for p in list_of_polarizations]
            except:
                print('WARNING could not guess the list of input polarizations. Considering it'
                      f' as {list_of_polarizations}. Please use the parameter'
                      ' --list-of-polarizations to inform the correct order')
                if self.input_raster:
                    list_of_polarizations = \
                        ['HH', 'HV', 'VH', 'VV'][0:input_raster_obj.nbands]
                else:
                    nbands_orig = nbands
                    if self.flag_symmetrize:
                        nbands_orig += 1
                    list_of_polarizations = \
                        ['HH', 'HV', 'VH', 'VV'][0:nbands_orig]

        print('list of input polarizations:', list_of_polarizations)

        if self.flag_symmetrize and 'VH' in list_of_polarizations:
            list_of_polarizations.remove('VH')
            print('list of processed polarizations:', list_of_polarizations)
        elif self.flag_symmetrize:
            vh_band = nbands - 1
            print('WARNING could not guess the VH polarization band in RSLC'
                  f' file. Considering it as RSLC band {vh_band}')
            del list_of_polarizations[vh_band]

            print('list of processed polarizations:', list_of_polarizations)

        if nbands <= 3:
            output_list_of_polarizations = ['HH', 'HV', 'VV'][0:nbands]
        else:
            output_list_of_polarizations = ['HH', 'HV', 'VH', 'VV']

        pol_map = []
        for b in range(nbands):
            ind = output_list_of_polarizations.index(
                list_of_polarizations[b]) + 1
            print('pol:', list_of_polarizations[b], ', C-matrix index: ', ind)
            pol_map.append(ind)

        for band_1 in range(nbands):
            for band_2 in range(nbands):
                out_band_1 = pol_map[band_1]
                out_band_2 = pol_map[band_2]

                if out_band_2 < out_band_1:
                    continue

                flag_apply_c3_cross_pol_factor = (nbands == 3 and
                                                  (out_band_1 == 2 or
                                                   out_band_2 == 2))

                # save diag terms
                if band_2 == band_1:
                    image = np.copy(image_obj.get_image(band=band_1))
                    ind = plant.isnan(image)
                    if flag_apply_c3_cross_pol_factor:
                        print(f'Applying 2x factor to C3 term:'
                              f' C{out_band_1}{out_band_2}')
                        image *= 2
                    image[ind] = PSP_NULL
                    c_term_file = os.path.join(self.covariance_matrix,
                                               f'C{out_band_1}{out_band_2}'
                                               '.bin')
                    plant.save_image(image, c_term_file, force=self.force)
                    plant.append_output_file(c_term_file)
                    continue

                image = image_off_diag_obj.get_image(band=band_index)
                band_index += 1

                # save off-diag real part
                c_term_file = os.path.join(self.covariance_matrix,
                                           f'C{out_band_1}{out_band_2}_real'
                                           '.bin')

                image_float = np.copy(np.real(image))
                ind = plant.isnan(image_float)
                if flag_apply_c3_cross_pol_factor:
                    print(f'Applying sqrt(2) factor to C3 term:'
                          ' C{out_band_1}{out_band_2}_real')
                    image_float *= np.sqrt(2)
                image_float[ind] = PSP_NULL

                plant.save_image(image_float, c_term_file, force=self.force)
                plant.append_output_file(c_term_file)

                # save off-diag imaginary part
                c_term_file = os.path.join(self.covariance_matrix,
                                           f'C{out_band_1}{out_band_2}_imag'
                                           '.bin')

                image_float = np.copy(np.imag(image))
                ind = plant.isnan(image_float)
                if flag_apply_c3_cross_pol_factor:
                    print('Applying sqrt(2) factor to C3 term:'
                          f' C{out_band_1}{out_band_2}_imag')
                    image_float *= np.sqrt(2)
                image_float[ind] = PSP_NULL

                plant.save_image(image_float, c_term_file,
                                 force=self.force)
                plant.append_output_file(c_term_file)

        config_file = os.path.join(self.covariance_matrix, 'config.txt')
        plant.create_config_txt(config_file,
                                width=width,
                                length=length,
                                force=self.force)

    def _get_radar_grid(self, slc_obj, frequency_str):
        radar_grid = slc_obj.getRadarGrid(frequency_str)

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
            print('cropping radar grid...')
            print('    before:', radar_grid.length, radar_grid.width)
            radar_grid = radar_grid.offset_and_resize(
                y0, x0, length, width)
            print('    after:', radar_grid.length, radar_grid.width)

        if (self.nlooks_az > 1 or self.nlooks_rg > 1):
            print('multilooking radar grid...')
            print('    before:', radar_grid.length, radar_grid.width)
            radar_grid_ml = radar_grid.multilook(self.nlooks_az,
                                                 self.nlooks_rg)
            print('    after:', radar_grid_ml.length, radar_grid_ml.width)
        else:
            radar_grid_ml = radar_grid

        return radar_grid_ml

    def _symmetrize_cross_pols(self, hv_ref, vh_ref):
        print(f'Symmetrizing: {hv_ref} and {vh_ref}')
        hv_raster_obj = isce3.io.Raster(hv_ref)
        vh_raster_obj = isce3.io.Raster(vh_ref)
        width = hv_raster_obj.width
        length = hv_raster_obj.length
        gdal_dtype = hv_raster_obj.datatype()
        temp_symmetrized_file = plant.get_temporary_file(
            append=True, suffix='_symmetrized', ext='tif')
        print('*** temporary symmetrized file:'
              f' {temp_symmetrized_file}')
        symmetrized_hv_raster_obj = self._get_raster(
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
        with plant.PlantIndent():
            output_band = 0
            for b in range(image_obj.nbands):
                band = image_obj.get_band(band=b)
                if (self.flag_symmetrize and
                        ((vh_band is not None and vh_band == b) or
                         (band.name is not None and
                          band.name.upper() == 'VH'))):
                    print('*** skipping VH')
                    continue
                if (self.flag_symmetrize and
                        ((hv_band is not None and hv_band == b) or
                         (band.name is not None and
                          band.name.upper() == 'HV'))):
                    symmetrized_hv_obj = self.read_image(temp_symmetrized_file)
                    symmetrized_band = symmetrized_hv_obj.band
                    image_obj.set_band(symmetrized_band, band=output_band)

                    symmetrized_band_out = image_obj.get_band(band=output_band)
                    output_band += 1
                    print('*** skipping HV')
                    print('*** reading symmetrized HV')
                    continue
                print(f'*** adding {band} to VRT file')
                image_obj.set_band(band, band=output_band)
                output_band += 1
            if self.flag_symmetrize:
                image_obj.set_nbands(image_obj.nbands - 1,
                                     realize_changes=False)
            self.save_image(image_obj, temp_file, force=True,
                            output_format=output_format)

    def _get_input_raster_from_nisar_slc(self, input_raster):
        if self.input_key and self.input_key == 'B':
            frequency_str = 'B'
        else:
            frequency_str = 'A'
        if input_raster is not None:
            if self.flag_transform_input_raster is not False:
                flag_apply_transformation = \
                    self.plant_transform_obj.flag_apply_transformation()
                image_obj = self.read_image(input_raster)
            else:
                flag_apply_transformation = False
                image_obj = plant.read_image(input_raster)
            if flag_apply_transformation:
                temp_file = plant.get_temporary_file(append=True,
                                                     ext='vrt')
                # image_obj = self.read_image(input_raster)
                for b in range(image_obj.nbands):
                    band = image_obj.get_band(band=b)
                    image_obj.set_band(band, band=b)
                self.print(f'*** creating temporary file: {temp_file}')
                self.save_image(image_obj, temp_file, force=True,
                                output_format='VRT')
                input_raster = temp_file

            if self.flag_symmetrize and self.symmetrize_bands is None:
                self.print('ERROR symmetrization option with input raster'
                           ' requires the parameter --symmetrize-bands')
                return
            elif self.flag_symmetrize:
                hv_band = self.symmetrize_bands[0]
                vh_band = self.symmetrize_bands[1]

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

                # ideally the output format would be a VRT, but currently
                # there's a bug in gdalbuildvrt that prevents using
                # specific bands from different inputs rasters
                self._get_symmetrized_input_raster(
                    image_obj, temp_file, temp_symmetrized_file,
                    hv_band=hv_band, vh_band=vh_band,
                    output_format='TIFF')
                input_raster = temp_file

        else:
            if self.flag_symmetrize:
                freq_group = ('//science/LSAR/RSLC/swaths/'
                              f'frequency{frequency_str}')
                hv_ref = f'HDF5:{self.input_file}:{freq_group}/HV'
                vh_ref = f'HDF5:{self.input_file}:{freq_group}/VH'
                flag_error = False
                try:
                    temp_symmetrized_file = self._symmetrize_cross_pols(
                        hv_ref, vh_ref)
                except:
                    flag_error = True

                if flag_error:
                    freq_group = ('//science/LSAR/SLC/swaths/'
                                  f'frequency{frequency_str}')
                    hv_ref = f'HDF5:{self.input_file}:{freq_group}/HV'
                    vh_ref = f'HDF5:{self.input_file}:{freq_group}/VH'
                    temp_symmetrized_file = self._symmetrize_cross_pols(
                        hv_ref, vh_ref)

            else:
                temp_symmetrized_file = None

            raster_file = f'NISAR:{self.input_file}:{frequency_str}'
            temp_file = plant.get_temporary_file(append=True,
                                                 ext='vrt')
            self.print(f'*** creating temporary file: {temp_file}')
            image_obj = self.read_image(raster_file)

            self._get_symmetrized_input_raster(
                image_obj, temp_file, temp_symmetrized_file,
                output_format='VRT')
            input_raster = temp_file

        ret_dict = {}
        ret_dict['input_raster'] = input_raster
        ret_dict['image_obj'] = image_obj
        return ret_dict

    def _get_doppler(self, slc_obj):
        # product, frequency_str
        if self.native_doppler:
            print('*** native dop')
            doppler = slc_obj.getDopplerCentroid()
            doppler.bounds_error = False
        else:
            # Make a zero-Doppler LUT
            print('*** zero dop')
            doppler = isce3.core.LUT2d()
        return doppler

    def _update_geogrid(self, proj_win_from_dem=None):

        if self.lon_arr is None:
            self.lon_arr = [proj_win_from_dem[0], proj_win_from_dem[2]]
        if self.lat_arr is None:
            self.lat_arr = [proj_win_from_dem[3], proj_win_from_dem[1]]

        if self.step_lat is None:
            self.step_lat = np.nan
        if self.step_lon is None:
            self.step_lon = np.nan

        if plant.isnan(self.step_lat) and self.step_lat > 0:
            self.lat_arr = [self.lat_arr[1], self.lat_arr[0]]
            self.step_lat = -self.step_lat

        if plant.isnan(self.lat_size) and plant.isvalid(self.step_lat):
            self.lat_size = (self.lat_arr[1] -
                             self.lat_arr[0])/self.step_lat

        if plant.isnan(self.lon_size) and plant.isvalid(self.step_lon):
            self.lon_size = (self.lon_arr[1] -
                             self.lon_arr[0])/self.step_lon


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Geocode(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
