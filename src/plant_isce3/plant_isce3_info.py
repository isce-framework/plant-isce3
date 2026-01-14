#!/usr/bin/env python3

import plant
import plant_isce3
import numpy as np
from osgeo import osr
from plant_isce3.readers import open_product

IDENTIFICATION_DICT = {
    'absoluteOrbitNumber':
        ['absolute orbit number', 1, int],
    'trackNumber':
        ['track number', 1, int],
    'frameNumber':
        ['frame number', 1, int],
    'granuleId':
        ['granule ID', None, str],
    'productVersion':
        ['product version', None, str],
    'processingDateTime':
        ['processing date time', None, str],
    'platformName':
        ['platform name', None, str],
    'isDithered':
        ['is dithered', None, str],
    'isMixedMode':
        ['is mixed mode', None, str],
    'isFullFrame':
        ['is full frame', None, str],
    'compositeReleaseId':
        ['composite release ID (CRID)', None, str],
    'isJointObservation':
        ['is joint observation', None, str]
}

FREQ_SWATH_DICT = {
    'slantRangeSpacing':
        ['slant-range spacing [m]', 1, float],
    'sceneCenterAlongTrackSpacing':
        ['scene-center along-track spacing [m]', 1, float],
    'sceneCenterGroundRangeSpacing':
        ['scene-center ground-range spacing [m]', 1, float],
    'processedRangeBandwidth':
        ['processed range bandwidth [MHz]', 1e-6, float],
    'acquiredRangeBandwidth':
        ['acquired range bandwidth [Hz]', 1e-6, float],
    'processedAzimuthBandwidth':
        ['processed azimuth bandwidth [Hz]', 1, float],
}


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            flag_debug=1,
                            input_files=1)

    plant_isce3.add_arguments(parser,
                              burst_ids=1,
                              epsg=1,
                              orbit_files=1)

    parser.add_argument('--all',
                        dest='show_all',
                        action='store_true',
                        help='Show all information (except stats)')

    return parser


class PlantIsce3Info(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        for i, self.input_file in enumerate(self.input_files):
            print(f'## input {i + 1}:', self.input_file)
            plant_product_obj = self.load_product(verbose=False)
            with plant.PlantIndent():
                sensor_name = plant_product_obj.sensor_name
                if (sensor_name == 'Sentinel-1'):
                    self._print_sentinel1_product_info(plant_product_obj)
                elif (sensor_name == 'NISAR'):
                    self._print_nisar_product_info()
                else:
                    print(f'ERROR unsupported product "{sensor_name}"')
                    return

    def _print_sentinel1_product_info(self, plant_product_obj):
        print('## Sentinel-1 product')
        pol_list = None
        burst_id_dict = {}

        print('## burst(s):')
        for burst_pol_dict in plant_product_obj.burst_dict.values():
            if pol_list is None:
                pol_list = list(burst_pol_dict.keys())
            burst = burst_pol_dict[pol_list[0]]
            burst_id = str(burst.burst_id)
            iw_index = burst_id.upper().find('IW')
            if iw_index < 0:
                print(f'ERROR invalid burst ID: {burst_id}')
            subswath_number = int(burst_id[iw_index + 2])
            iw_key = f'IW{subswath_number}'
            if iw_key not in burst_id_dict.keys():
                burst_id_dict[iw_key] = []
            burst_id_dict[iw_key].append(burst_id)
            with plant.PlantIndent():
                y = burst.center.y
                x = burst.center.x
                epsg = plant_isce3.point2epsg(x, y)
                y_str = plant.format_number(y, sigfigs=4)
                x_str = plant.format_number(x, sigfigs=4)
                print(burst_id)
                with plant.PlantIndent():
                    print(f'center pos. (Y, X): ({y_str}, {x_str})')
                    print(f'EPSG: {epsg}')

        print('## polarizations:', pol_list)
        for key, value in burst_id_dict.items():
            print(f'## burst(s) in subswath {key}:', value)

    def verify_lut_horizontal_extents(
            self, product_type,
            h5_obj, lut_group_path, lut_name,
            y_coordinates_str, x_coordinates_str,
            y0, yf, dy, x0, xf, dx,
            coordinate_y_descr,
            coordinate_x_descr):

        if (f'{lut_group_path}/{y_coordinates_str}' not in
                h5_obj[lut_group_path]):
            lut_group_parent_path = '/'.join(lut_group_path.split('/')[:-1])
            print(f'*** path {y_coordinates_str} not found, retrying it with'
                  f' {lut_group_parent_path}/{y_coordinates_str}')
            y_coordinates = h5_obj[
                f'{lut_group_parent_path}/{y_coordinates_str}'][()]
        else:
            y_coordinates = h5_obj[
                f'{lut_group_path}/{y_coordinates_str}'][()]
        x_coordinates = h5_obj[
            f'{lut_group_path}/{x_coordinates_str}'][()]

        lut_y0 = y_coordinates[0]
        lut_yf = y_coordinates[-1]
        lut_dy = y_coordinates[1] - y_coordinates[0]

        lut_x0 = x_coordinates[0]
        lut_xf = x_coordinates[-1]
        lut_dx = x_coordinates[1] - x_coordinates[0]

        lut_y0_index = (lut_y0 - y0) / lut_dy
        lut_yf_index = (lut_yf - yf) / lut_dy
        lut_x0_index = (lut_x0 - x0) / lut_dx
        lut_xf_index = (lut_xf - xf) / lut_dx

        lut_y0_status = '[ OK ]' if lut_dy / \
            abs(lut_dy) * lut_y0_index < 0 else '[FAIL]'
        lut_yf_status = '[ OK ]' if lut_dy / \
            abs(lut_dy) * lut_yf_index > 0 else '[FAIL]'

        lut_x0_status = '[ OK ]' if lut_x0_index < 0 else '[FAIL]'
        lut_xf_status = '[ OK ]' if lut_xf_index > 0 else '[FAIL]'

        print(f'{lut_name}:')

        with plant.PlantIndent():

            if (product_type != 'RSLC' or
                    'noiseEquivalentBackscatter' not in lut_group_path):
                print(f'{lut_y0_status} start {coordinate_y_descr}'
                      f' (lut_y0 - y0) / lut_dy = lut_y0_index => '
                      f' ({lut_y0} - {y0}) / {lut_dy} ='
                      f' {lut_y0_index}')
                print(f'{lut_yf_status} end {coordinate_y_descr}'
                      f' (lut_yf - yf) / lut_dy = lut_yf_index => '
                      f' ({lut_yf} - {yf}) / {lut_dy} ='
                      f' {lut_yf_index}')

            print(f'{lut_x0_status} start {coordinate_x_descr}'
                  f' (lut_x0 - x0) / lut_dx = lut_x0_index => '
                  f' ({lut_x0} - {x0}) / {lut_dx} ='
                  f' {lut_x0_index}')
            print(f'{lut_xf_status} end {coordinate_x_descr}'
                  f' (lut_xf - xf) / lut_dx = lut_xf_index => '
                  f' ({lut_xf} - {xf}) / {lut_dx} ='
                  f' {lut_xf_index}')

    def _print_nisar_product_info(self):
        import shapely.wkt
        nisar_product_obj = open_product(self.input_file)

        print('## NISAR product')
        print('## product type:', nisar_product_obj.productType)
        print('## SAR band:', nisar_product_obj.sarBand)
        print('## level:', nisar_product_obj.getProductLevel())

        print('## frequencies/polarizations:')
        freq_pol_dict = nisar_product_obj.polarizations
        with plant.PlantIndent():
            for freq, pol_list in freq_pol_dict.items():
                print(f'{freq}: {pol_list}')

        h5_obj = plant.h5py_file_wrapper(self.input_file, 'r', swmr=True)

        metadata_path = nisar_product_obj.MetadataPath

        if nisar_product_obj.productType == 'GCOV':

            print('## covariance terms:')
            freq_pol_dict = nisar_product_obj.covarianceTerms
            with plant.PlantIndent():
                for frequency, cov_terms_list in freq_pol_dict.items():
                    print(f'{frequency}: {cov_terms_list}')

            is_full_covariance = h5_obj[
                f'{metadata_path}/processingInformation/parameters/'
                'isFullCovariance'][()]
            if not isinstance(is_full_covariance, str):
                is_full_covariance = is_full_covariance.decode()
            print('## full covariance (True/False):', is_full_covariance)

        if nisar_product_obj.productType == 'STATIC':
            swaths_base_path = f'{metadata_path}/radarGridParameters'
        elif nisar_product_obj.getProductLevel() == 'L2':
            swaths_base_path = f'{metadata_path}/sourceData/swaths/'
            image_path = nisar_product_obj.GridPath
        else:
            swaths_base_path = nisar_product_obj.SwathPath
            image_path = nisar_product_obj.SwathPath

        if nisar_product_obj.productType != 'RRSD':
            pri_path = (f'{swaths_base_path}/zeroDopplerTimeSpacing')
            pri = h5_obj[pri_path][()]

        flag_color = self.getattr2('flag_color_text') is not False

        print('## other parameters:')
        with plant.PlantIndent():
            if nisar_product_obj.productType != 'RRSD':
                print(plant.bcolors.Yellow +
                      f'pulse repetition interval (PRI) [ms]: {pri * 1e3}' +
                      plant.bcolors.ColorOff)
                print(plant.bcolors.Yellow +
                      f'pulse repetition frequency (PRF) [Hz]: {1. / pri}' +
                      plant.bcolors.ColorOff)

            self.print_h5_parameters(h5_obj,
                                     nisar_product_obj.IdentificationPath,
                                     IDENTIFICATION_DICT,
                                     flag_color)

            for frequency in freq_pol_dict.keys():
                if nisar_product_obj.productType == 'RRSD':
                    continue

                print(f'## frequency {frequency}')

                freq_swaths_path = (f'{swaths_base_path}/'
                                    f'frequency{frequency}')

                if nisar_product_obj.productType == 'RSLC':
                    nominal_acquisition_prf_path = \
                        f'{freq_swaths_path}/nominalAcquisitionPRF'

                    nominal_acquisition_prf = \
                        h5_obj[nominal_acquisition_prf_path][()]

                with plant.PlantIndent():

                    if nisar_product_obj.productType == 'GCOV':
                        cov_terms_list = freq_pol_dict[frequency]
                        first_image_path = \
                            (f'{image_path}/frequency{frequency}/'
                             f'{cov_terms_list[0]}')
                    else:
                        first_image_path = \
                            (f'{image_path}/frequency{frequency}/'
                             f'{freq_pol_dict[frequency][0]}')
                    first_image_shape = h5_obj[first_image_path].shape

                    if flag_color:
                        print(plant.bcolors.Cyan +
                              'number of lines (azimuth lines):'
                              f' {first_image_shape[0]}' +
                              plant.bcolors.ColorOff)
                        print(plant.bcolors.Cyan +
                              'number of samples (range bins):'
                              f' {first_image_shape[1]}' +
                              plant.bcolors.ColorOff)
                    else:
                        print('## number of lines (azimuth lines):'
                              f' {first_image_shape[0]}')
                        print('## number of samples (range bins):'
                              f' {first_image_shape[1]}')

                    if nisar_product_obj.productType == 'RSLC':
                        print('nominal acquisition PRF [Hz]:'
                              f' {nominal_acquisition_prf}')

                    self.print_h5_parameters(h5_obj, freq_swaths_path,
                                             FREQ_SWATH_DICT,
                                             flag_color)

        if self.show_all and nisar_product_obj.productType == 'RSLC':
            print('## Image and look-up tables (LUTs) extents:')
            with plant.PlantIndent():

                for frequency in freq_pol_dict.keys():
                    print(f'## frequency {frequency}')

                    with plant.PlantIndent():

                        if nisar_product_obj.productType == 'RSLC':
                            image_group_path = (f'{swaths_base_path}/'
                                                f'frequency{frequency}')
                            y_coordinates_str = 'zeroDopplerTime'
                            x_coordinates_str = 'slantRange'
                            coordinate_y_descr = 'azimuth time [s]'
                            coordinate_x_descr = 'slant range [m]'
                            y_coordinates = h5_obj[
                                f'{swaths_base_path}/{y_coordinates_str}'][()]
                        else:

                            image_group_path = \
                                f'{image_path}/frequency{frequency}/'
                            y_coordinates_str = 'yCoordinates'
                            x_coordinates_str = 'xCoordinates'
                            coordinate_y_descr = 'Y coordinates [m]'
                            coordinate_x_descr = 'X coordinates [m]'

                            y_coordinates = h5_obj[
                                f'{image_group_path}/{y_coordinates_str}'][()]

                        x_coordinates = h5_obj[
                            f'{image_group_path}/{x_coordinates_str}'][()]

                        y0 = y_coordinates[0]
                        yf = y_coordinates[-1]
                        dy = y_coordinates[1] - y_coordinates[0]

                        x0 = x_coordinates[0]
                        xf = x_coordinates[-1]
                        dx = x_coordinates[1] - x_coordinates[0]

                        print('imagery:')
                        with plant.PlantIndent():
                            print(f'y: [{y0}, {yf}], dx: {dy}')
                            print(f'x: [{x0}, {xf}], dy: {dx}')

                        lut_list = [

                            [f'{metadata_path}/geolocationGrid/',
                             'geolocationGrid'],

                            [f'{metadata_path}/calibrationInformation/'
                             f'frequency{frequency}/elevationAntennaPattern',
                             f'frequency{frequency}/elevationAntennaPattern'],

                            [f'{metadata_path}/calibrationInformation/'
                             f'frequency{frequency}/'
                             'noiseEquivalentBackscatter',
                             f'frequency{frequency}/'
                             'noiseEquivalentBackscatter'],

                            [f'{metadata_path}/processingInformation/'
                             f'parameters/frequency{frequency}',
                             f'frequency{frequency}/processingInformation/'
                             'dopplerCentroid'],

                            [f'{metadata_path}/calibrationInformation/'
                             'geometry',
                             'geometry'],

                        ]

                        for lut_group_path, lut_name in lut_list:

                            self.verify_lut_horizontal_extents(
                                nisar_product_obj.productType,
                                h5_obj, lut_group_path, lut_name,
                                y_coordinates_str, x_coordinates_str,
                                y0, yf, dy, x0, xf, dx,
                                coordinate_y_descr,
                                coordinate_x_descr)

        h5_obj.close()

        polygon = nisar_product_obj.identification.boundingPolygon

        if (nisar_product_obj.productType == 'GCOV' or
                nisar_product_obj.productType == 'GSLC'):
            for freq, pol_list in freq_pol_dict.items():
                print(f'## geogrid frequency {freq}')
                with plant.PlantIndent():
                    image_obj = self.read_image(f'NISAR:{self.input_file}:'
                                                f'{freq}')
                    plant_geogrid_obj = plant.get_coordinates(
                        image_obj=image_obj)
                    plant_geogrid_obj.print()

        print('## bounding polygon:')
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
                bbox = plant.get_bbox(x0=x0, xf=xf, y0=y0, yf=yf)
                coord_str = ('PLAnT bbox argument: -b %.16f %.16f %.16f %.16f'
                             % (bbox[0], bbox[1], bbox[2], bbox[3]))
                print(coord_str)

            if self.epsg is None:
                zones_list = []
                for lat in [y0, yf]:
                    for lon in [x0, xf]:
                        zones_list.append(plant_isce3.point2epsg(lon, lat))
                vals, counts = np.unique(zones_list, return_counts=True)
                self.epsg = int(vals[np.argmax(counts)])
                print('closest projection EPSG code supported by NISAR:',
                      self.epsg)

            if self.epsg is not None:
                y_min = np.nan
                y_max = np.nan
                x_min = np.nan
                x_max = np.nan
                for lat in [y0, yf]:
                    for lon in [x0, xf]:
                        y, x = lat_lon_to_projected(lat, lon, self.epsg)
                        if plant.isnan(y_min) or y < y_min:
                            y_min = y
                        if plant.isnan(y_max) or y > y_max:
                            y_max = y
                        if plant.isnan(x_min) or x < x_min:
                            x_min = x
                        if plant.isnan(x_max) or x > x_max:
                            x_max = x

                projected_bbox = plant.get_bbox(x0=x_min, xf=x_max, y0=y_max,
                                                yf=y_min)
                coord_str = ('PLAnT bbox argument: -b %.0f %.0f %.0f %.0f'
                             % (projected_bbox[0], projected_bbox[1],
                                projected_bbox[2], projected_bbox[3]))

                print(f'EPSG {self.epsg} coordinates:')
                with plant.PlantIndent():
                    print('min Y:', y_min)
                    print('min X:', x_min)
                    print('max Y:', y_max)
                    print('max X:', x_max)
                    print(coord_str)

    def print_h5_parameters(self, h5_obj, h5path, dict_datasets_text,
                            flag_color):
        for dataset, (text, factor, dtype) in dict_datasets_text.items():
            h5_path = f'{h5path}/{dataset}'
            if h5_path not in h5_obj:
                continue
            h5_dataset = h5_obj[h5_path]
            if dtype == str:
                value = h5_dataset[()]
                if not isinstance(value, str):
                    value = value.decode()
            else:
                value = dtype(factor * h5_dataset[()])

            if (flag_color and
                    (text.startswith('slant-range spacing [m]') or
                     text.startswith('scene-center along-track spacing [m]'))):
                print(plant.bcolors.Green + f'{text}: {value}' +
                      plant.bcolors.ColorOff)
            elif (flag_color and
                    text.startswith('processed range bandwidth') or
                    text.startswith('processed azimuth bandwidth')):
                print(plant.bcolors.Purple + f'{text}: {value}' +
                      plant.bcolors.ColorOff)
            elif (text.startswith('slant-range spacing [m]') or
                    text.startswith('scene-center along-track spacing [m]') or
                    text.startswith('processed range bandwidth')):
                print(f'## {text}: {value}')
            else:
                print(f'{text}: {value}')


def lat_lon_to_projected(north, east, epsg):
    osr.UseExceptions()

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

    transformation = osr.CoordinateTransformation(wgs84_coordinate_system,
                                                  projected_coordinate_system)
    x, y, _ = transformation.TransformPoint(float(east), float(north), 0)
    return (y, x)


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Info(parser, argv)
        ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
