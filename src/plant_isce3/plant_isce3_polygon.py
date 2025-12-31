#!/usr/bin/env python3

import os
import plant
import plant_isce3
import numpy as np
from osgeo import gdal
import random
import isce3
from plant_isce3.readers import SLC, open_product


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=2,
                            default_options=1,
                            multilook=1,
                            output_file=1)

    plant_isce3.add_arguments(parser,
                              abs_cal_factor=1,
                              input_raster=1,
                              native_doppler_grid=1)

    parser.add_argument('--exponent',
                        dest='exponent',
                        type=int,
                        help='Exponent. Choices: 0, 1, and 2')

    parser.add_argument('--add-off-diag-terms',
                        '--add-off-diagonal-terms',
                        '--include-off-diag-terms',
                        '--include-off-diagonal-terms',
                        dest='flag_add_off_diag_terms',
                        action='store_true',
                        help='Include off-diagonal terms.')

    parser.add_argument('--save-radargrid-data',
                        '--save-radar-data',
                        action='store_true',
                        dest='save_radargrid_data',
                        help='Sava radar-grid data')

    parser.add_argument('--save-rtc',
                        '--save-rtc-area-factor',
                        action='store_true',
                        dest='save_rtc',
                        help='Save RTC area factor')

    parser.add_argument('--save-weights',
                        action='store_true',
                        dest='save_weights',
                        help='Save area projection weights')

    parser_output_mode = parser.add_mutually_exclusive_group()
    parser_output_mode.add_argument('--area',
                                    action='store_true',
                                    dest='output_mode_area',
                                    help='Use area mode')

    parser_output_mode.add_argument('--area-gamma-naught',
                                    action='store_true',
                                    dest='output_mode_area_gamma_naught',
                                    help='Use area mode and apply radiometric'
                                    ' terrain correction')

    parser.add_argument('--upsampling',
                        dest='geogrid_upsampling',
                        type=float,
                        help='Geogrid upsample factor.')

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

    parser.add_argument('--out-nlooks',
                        dest='out_nlooks',
                        type=str,
                        help='Output nlooks file')

    return parser


class PlantIsce3Polygon(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        if not self.plant_transform_obj.geo_polygon:
            self.print('ERROR one the following argument is required:'
                       f' {self.plant_transform_obj.geo_polygon}')
            return

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        plant_product_obj = self.load_product()

        orbit = plant_product_obj.get_orbit()
        doppler = plant_product_obj.get_grid_doppler()

        self.tec_file = plant_product_obj.get_tec_file()

        if (plant_product_obj.sensor_name == 'Sentinel-1'):

            input_raster = plant_product_obj.get_sentinel_1_input_raster(
                self.input_raster,
                flag_transform_input_raster=self.flag_transform_input_raster)

        else:
            input_raster = self.get_input_raster_from_nisar_product(
                plant_product_obj=plant_product_obj)

        input_raster_obj = plant_isce3.get_isce3_raster(input_raster)

        ret_dict = self.load_product()
        radar_grid_ml = ret_dict['radar_grid_ml']
        orbit = ret_dict['orbit']
        doppler = ret_dict['grid_doppler']

        ellipsoid = isce3.core.Ellipsoid()

        if input_raster_obj.datatype() == 6:
            GeocodePolygon = isce3.geocode.GeocodePolygonFloat32
        elif input_raster_obj.datatype() == 7:
            GeocodePolygon = isce3.geocode.GeocodePolygonFloat64
        elif input_raster_obj.datatype() == 10:
            GeocodePolygon = isce3.geocode.GeocodePolygonCFloat32
        elif input_raster_obj.datatype() == 11:
            GeocodePolygon = isce3.geocode.GeocodePolygonCFloat64
        else:
            raise NotImplementedError('Unsupported raster type for geocoding')

        dem_raster = plant_isce3.get_isce3_raster(self.dem_file)
        if dem_raster.get_epsg() == 0 or dem_raster.get_epsg() < -9000:
            print(f'WARNING invalid DEM EPSG: {dem_raster.get_epsg()}')
            print('Updating DEM EPSG to 4326...')
            dem_raster.set_epsg(4326)

        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        input_dtype = input_raster_obj.datatype()
        print('*** exponent: ', self.exponent)
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

        print(f'*** input_raster.width: {input_raster_obj.width}')
        print(f'*** input_raster.length: {input_raster_obj.length}')

        kwargs = {}
        if self.output_mode_area_gamma_naught:
            flag_error = False
            try:
                output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION_GAMMA_NAUGHT
            except BaseException:
                flag_error = True
            if flag_error:
                try:
                    output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION_WITH_RTC
                    flag_error = False
                except BaseException:
                    pass
            if flag_error:
                output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION
                kwargs['flag_apply_rtc'] = True
        else:
            output_mode = isce3.geocode.GeocodeOutputMode.AREA_PROJECTION

        flag_sigma = (self.input_terrain_radiometry is not None and
                      ('sigma-inc' in self.input_terrain_radiometry or
                       'sigma-el' in self.input_terrain_radiometry))

        if flag_sigma:
            kwargs['input_terrain_radiometry'] = \
                isce3.geometry.RtcInputTerrainRadiometry.SIGMA_NAUGHT_ELLIPSOID
        else:
            kwargs['input_terrain_radiometry'] = \
                isce3.geometry.RtcInputTerrainRadiometry.BETA_NAUGHT

        if (self.output_terrain_radiometry is not None and
                'beta' in self.output_terrain_radiometry.lower()):
            output_terrain_radiometry = isce3.geometry.RtcOutputTerrainRadiometry.BETA_NAUGHT_TEST
        elif (self.output_terrain_radiometry is not None and
              'sigma' in self.output_terrain_radiometry.lower()):
            output_terrain_radiometry = isce3.geometry.RtcOutputTerrainRadiometry.SIGMA_NAUGHT
        else:
            output_terrain_radiometry = isce3.geometry.RtcOutputTerrainRadiometry.GAMMA_NAUGHT
        kwargs['output_terrain_radiometry'] = output_terrain_radiometry

        if self.geogrid_upsampling is None:
            self.geogrid_upsampling = 1

        if self.rtc_min_value_db is not None:
            kwargs['rtc_min_value_db'] = self.rtc_min_value_db

        if self.abs_cal_factor is not None:
            kwargs['abs_cal_factor'] = self.abs_cal_factor

        kwargs['radargrid_nlooks'] = (self.nlooks_az *
                                      self.nlooks_rg)

        print('*** geo-polygon:',
              self.plant_transform_obj.geo_polygon)
        parsed_polygon = plant.parse_polygon_from_file(
            self.plant_transform_obj.geo_polygon,
            verbose=self.verbose)
        if parsed_polygon is None:
            return
        y_vect_list, x_vect_list = parsed_polygon
        if len(y_vect_list) == 0:
            self.print('ERROR polygon Y-coordinate list is empty')
            return

        if len(y_vect_list) == 0:
            self.print('ERROR polygon Y-coordinate list is empty')
            return

        n_polygons = np.min([len(y_vect_list), len(x_vect_list)])

        result_list = [None] * n_polygons

        if self.out_nlooks:
            nlooks_list = [np.nan] * n_polygons

        for i, (y_vect, x_vect) in enumerate(
                zip(y_vect_list, x_vect_list)):
            self.print(f'Processing polygon {i + 1}:')
            with plant.PlantIndent():
                self.print(f'Y-vect: {y_vect}')
                self.print(f'X-vect: {x_vect}')
                temp_file = plant.get_temporary_file(
                    suffix=f'polygon_{i + 1}_temp_{random.random()}',
                    append=True)
                plant.append_temporary_file(temp_file)
                out_polygon_raster_obj = plant_isce3.get_isce3_raster(
                    temp_file,
                    nbands,
                    1,
                    1,
                    output_dtype,
                    'ENVI')

                out_off_diag_terms_obj = None
                temp_off_diag_file = None
                nbands_off_diag_terms = None
                if self.flag_add_off_diag_terms:
                    nbands_off_diag_terms = int((nbands**2 - nbands) / 2)
                    print('nbands_off_diag_terms: ', nbands_off_diag_terms)
                    if nbands_off_diag_terms > 0:
                        temp_off_diag_file = plant.get_temporary_file(
                            suffix=f'polygon_{i + 1}_temp_{random.random()}',
                            append=True)
                        plant.append_temporary_file(temp_off_diag_file)
                        out_off_diag_terms_obj = plant_isce3.get_isce3_raster(
                            temp_off_diag_file,
                            nbands_off_diag_terms,
                            1,
                            1,
                            gdal.GDT_CFloat32,
                            'ENVI')
                try:
                    polygon_obj = GeocodePolygon(
                        x_vect,
                        y_vect,
                        radar_grid_ml,
                        orbit,
                        ellipsoid,
                        doppler,
                        dem_raster)
                except BaseException:
                    error_message = plant.get_error_message()
                    self.print('ERROR there was an error processing'
                               f' polygon {i + 1}: ' +
                               error_message)
                    if not self.flag_add_off_diag_terms:
                        result_list[i] = np.full((1, nbands), np.nan)
                    else:
                        result_list[i] = np.full((nbands, nbands), np.nan,
                                                 dtype=np.complex64)
                    continue

                width = polygon_obj.xsize
                length = polygon_obj.ysize
                print(f'*** cropped radar grid dimensions: {width}x{length}')

                if self.save_radargrid_data:
                    radargrid_data_filename = f'polygon_{i + 1}_data.bin'
                    output_radargrid_data_obj = plant_isce3.get_isce3_raster(
                        radargrid_data_filename,
                        width,
                        length,
                        nbands,
                        output_dtype,
                        "ENVI")
                    kwargs['output_radargrid_data'] = output_radargrid_data_obj
                    plant.append_output_file(radargrid_data_filename)

                if self.save_rtc:
                    rtc_filename = f'polygon_{i + 1}_rtc.bin'
                    output_rtc_obj = plant_isce3.get_isce3_raster(
                        rtc_filename,
                        width,
                        length,
                        1,
                        gdal.GDT_Float32,
                        "ENVI")
                    kwargs['output_rtc'] = output_rtc_obj
                    plant.append_output_file(rtc_filename)

                if self.save_weights:
                    weights_filename = f'polygon_{i + 1}_weights.bin'
                    output_weights_obj = plant_isce3.get_isce3_raster(
                        weights_filename,
                        width,
                        length,
                        1,
                        gdal.GDT_Float64,
                        "ENVI")
                    kwargs['output_weights'] = output_weights_obj
                    plant.append_output_file(weights_filename)

                flag_error = False
                try:
                    polygon_obj.get_polygon_mean(
                        radar_grid_ml,
                        doppler,
                        input_raster_obj,
                        out_polygon_raster_obj,
                        dem_raster,
                        output_off_diag_terms=out_off_diag_terms_obj,
                        exponent=self.exponent,
                        output_mode=output_mode,
                        geogrid_upsampling=self.geogrid_upsampling,
                        **kwargs)
                except BaseException:
                    flag_error = True
                if flag_error:
                    try:
                        polygon_obj.get_polygon_mean(
                            radar_grid_ml,
                            doppler,
                            input_raster_obj,
                            out_polygon_raster_obj,
                            dem_raster,
                            output_off_diag_terms=out_off_diag_terms_obj,
                            exponent=self.exponent,
                            geogrid_upsampling=self.geogrid_upsampling,
                            **kwargs)
                    except BaseException:
                        error_message = plant.get_error_message()
                        self.print(f'There was an error processing'
                                   f' polygon {i + 1}: ' +
                                   error_message)
                        if not self.flag_add_off_diag_terms:
                            result_list[i] = np.full((1, nbands), np.nan)
                        else:
                            result_list[i] = np.full((nbands, nbands), np.nan,
                                                     dtype=np.complex64)
                        continue

                if self.save_radargrid_data:
                    output_radargrid_data_obj = kwargs['output_radargrid_data']
                    del output_radargrid_data_obj

                if self.save_rtc:
                    output_rtc_obj = kwargs['output_rtc']
                    del output_rtc_obj

                if self.save_weights:
                    output_weights_obj = kwargs['output_weights']
                    del output_weights_obj

                if self.out_nlooks:
                    nlooks = polygon_obj.out_nlooks
                    print('*** nlooks: ', nlooks)
                    nlooks_list[i] = nlooks
                del out_polygon_raster_obj
                mean_value_image = plant.read_image(temp_file).image
                print('*** mean_value_image:', mean_value_image)
                mean_value = np.asarray(mean_value_image[0]).reshape(1, nbands)
                print('*** mean_value returned: ', mean_value)
                print('*** mean_value.shape:', mean_value.shape)

                if self.flag_add_off_diag_terms:
                    del out_off_diag_terms_obj
                    mean_off_value = np.asarray(plant.read_image(
                        temp_off_diag_file).image[0])
                    print('*** mean_value off-diag returned: ', mean_off_value)

                    cov_matrix = np.full((nbands, nbands), np.nan,
                                         dtype=np.complex64)
                    band_index = 0
                    for band_1 in range(nbands):
                        for band_2 in range(nbands):
                            if band_1 == band_2:
                                cov_matrix[band_1,
                                           band_2] = mean_value[0, band_1]
                            elif band_1 < band_2:
                                cov_matrix[band_1,
                                           band_2] = mean_off_value[band_index]
                                band_index += 1
                            else:
                                cov_matrix[band_1, band_2] = np.conj(
                                    cov_matrix[band_2, band_1])
                    print('*** cov_matrix: ', cov_matrix)
                    mean_value = cov_matrix

            result_list[i] = mean_value

        del input_raster_obj

        if self.out_nlooks:
            self.save_image(nlooks_list, self.out_nlooks)

        self.save_image(result_list, self.output_file)
        plant.append_output_file(self.output_file)
        return result_list


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Polygon(parser, argv)
        ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
