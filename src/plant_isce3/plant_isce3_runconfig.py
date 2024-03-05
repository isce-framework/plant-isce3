#!/usr/bin/env python3

import os
import sys
import plant
from osgeo import gdal, osr
import numpy as np
import isce3
from nisar.products.readers import SLC, open_product
import nisar.workflows.helpers as helpers
import yamale
from ruamel.yaml import YAML


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
                            # multilook=1,
                            output_file=1)

    parser.add_argument('--runconfig',
                        dest='runconfig',
                        type=str,
                        help='Input runconfig.')

    parser.add_argument('--use-default-runconfig',
                        dest='use_default_runconfig',
                        default=False,
                        action='store_true',
                        help='Use default runconfig.')

    parser.add_argument('--tec',
                        '--tec-file',
                        dest='tec_file',
                        type=str,
                        help='Total electron content file.')

    parser.add_argument('--sas-output-file',
                        '--sas-of',
                        dest='sas_output_file',
                        type=str,
                        help='SAS output file.')

    parser.add_argument('--external-orbit',
                        '--external-orbit_file',
                        dest='external_orbit_file',
                        type=str,
                        help='Exterinal orbit file.')

    parser.add_argument('--workflow',
                        '--workflow-name',
                        dest='workflow_name',
                        type=str,
                        default='GCOV',
                        help='Workflow name')

    parser.add_argument('--epsg',
                        dest='epsg',
                        type=int,
                        help='EPSG code for output grids.')

    parser.add_argument('--snap-x',
                        dest='snap_x',
                        type=float,
                        help='X-coordinates snap.')

    parser.add_argument('--snap-y',
                        dest='snap_y',
                        type=float,
                        help='X-coordinates snap.')

    return parser


class PlantIsce3Runconfig(plant.PlantScript):

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

        ellipsoid = isce3.core.Ellipsoid()

        slc_obj = SLC(hdf5file=self.input_file)
        orbit = slc_obj.getOrbit()

        # Get radar grid
        radar_grid = self._get_radar_grid(slc_obj, frequency_str)

        geo = isce3.geocode.GeocodeFloat32()

        # init geocode members
        geo.orbit = orbit
        geo.ellipsoid = ellipsoid

        self._get_coordinates_from_h5_file(self.input_file)

        dem_raster = isce3.io.Raster(self.dem_file)

        lat_size = self.lat_size if plant.isvalid(self.lat_size) else -9999
        lon_size = self.lon_size if plant.isvalid(self.lon_size) else -9999
        y0_orig = self.lat_arr[1]
        x0_orig = self.lon_arr[0]

        # default pixel size
        # if self.step_lon is None and self.epsg != 4326:
        #     self.step_lon = 20

        if self.epsg == 4326 and not plant.isvalid(self.step_lon):
            self.step_lon = plant.m_to_deg_lon(30.)
        elif self.step_lon is None:
            self.step_lon = np.nan

        if self.epsg == 4326 and not plant.isvalid(self.step_lat):
            self.step_lat = plant.m_to_deg_lat(30.)
        elif self.step_lat is None:
            self.step_lat = np.nan

        # if self.step_lat is None and self.epsg != 4326:
        #     self.step_lat = -20
        # default snap values
        # if (self.snap_x is None and self.epsg != 4326 and
        #         self.workflow_name.upper() == 'GCOV'):
        #     self.snap_x = 100
        if self.snap_x is None and self.epsg != 4326:
            self.snap_x = 80
        # if (self.snap_y is None and self.epsg != 4326 and
        #         self.workflow_name.upper() == 'GCOV'):
        #     self.snap_y = 100
        if self.snap_y is None and self.epsg != 4326:
            self.snap_y = 80

        geo.geogrid(x0_orig,
                    y0_orig,
                    self.step_lon,
                    self.step_lat,
                    int(lon_size),
                    int(lat_size),
                    self.epsg)

        geo.update_geogrid(radar_grid, dem_raster)

        x0 = geo.geogrid_start_x
        y0 = geo.geogrid_start_y
        dx = geo.geogrid_spacing_x
        dy = geo.geogrid_spacing_y
        geogrid_width = abs(geo.geogrid_width)
        geogrid_length = abs(geo.geogrid_length)
        xf = x0 + dx * geogrid_width
        yf = y0 + dy * geogrid_length

        # get pixel spacing
        freq_a_dx = None
        freq_a_dy = None
        freq_b_dx = None
        freq_b_dy = None

        if plant.isvalid(self.step_lon) and 'A' in slc_obj.frequencies:
            freq_a_dx = self.step_lon
        if plant.isvalid(self.step_lon) and 'B' in slc_obj.frequencies:
            freq_b_dx = self.step_lon

        if plant.isvalid(self.step_lat) and 'A' in slc_obj.frequencies:
            freq_a_dy = self.step_lat
        if plant.isvalid(self.step_lat) and 'B' in slc_obj.frequencies:
            freq_b_dy = self.step_lat

        freq_a_dx, freq_a_dy, freq_b_dx, freq_b_dy = \
            self.get_pixel_spacing(freq_a_dx, freq_a_dy, freq_b_dx, freq_b_dy,
                                   slc_obj)

        if self.snap_x:
            x0 = snap_coord(x0, self.snap_x, 0, np.floor)
            xf = snap_coord(xf, self.snap_x, 0, np.ceil)

        if self.snap_y:
            yf = snap_coord(yf, self.snap_x, 0, np.floor)
            y0 = snap_coord(y0, self.snap_x, 0, np.ceil)

        if self.sas_output_file is None:
            self.sas_output_file = (f'output_{self.workflow_name.lower()}/'
                                    f'{self.workflow_name.lower()}.h5')

        print('=============================================================')
        self.print_runconfig(x0, y0, xf, yf, freq_a_dx, freq_a_dy, freq_b_dx,
                             freq_b_dy)
        print('==============================================================')

        if (os.path.isfile(self.output_file) or
                os.path.islink(self.output_file)):
            flag_update_file = self.overwrite_file_check(
                self.output_file)
            if not flag_update_file:
                return

        if not self.use_default_runconfig and not self.runconfig:

            if self.output_file:
                with open(self.output_file, 'w') as output_file_obj:
                    self.print_runconfig(x0, y0, xf, yf, freq_a_dx, freq_a_dy,
                                         freq_b_dx, freq_b_dy,
                                         file=output_file_obj)

            return

        default_runconfig_basename = f'{self.workflow_name.lower()}.yaml'
        if self.runconfig is None:
            self.runconfig = os.path.join(f'{helpers.WORKFLOW_SCRIPTS_DIR}',
                                          'defaults',
                                          default_runconfig_basename)
            print('default runconfing path:', self.runconfig)
        else:
            print('input runconfing path:', self.runconfig)

        # load default config
        parser = YAML(typ='safe')
        with open(self.runconfig, 'r') as f:
            cfg = parser.load(f)

            groups = cfg['runconfig']['groups']
            groups['input_file_group']['input_file_path'] = self.input_file
            groups['dynamic_ancillary_file_group']['dem_file'] = self.dem_file

            if self.tec_file:
                groups['dynamic_ancillary_file_group']['tec_file'] = \
                    self.tec_file
            if self.external_orbit_file:
                groups['dynamic_ancillary_file_group']['orbit_file'] = \
                    self.external_orbit_file

            groups['processing']['geocode']['output_epsg'] = int(self.epsg)

            if freq_a_dx is not None:
                groups['processing']['geocode'][
                    'output_posting']['A']['x_posting'] = float(freq_a_dx)
            if freq_a_dy is not None:
                groups['processing']['geocode'][
                    'output_posting']['A']['y_posting'] = float(freq_a_dy)

            if freq_b_dx is not None:
                groups['processing']['geocode'][
                    'output_posting']['B']['x_posting'] = float(freq_b_dx)
            if freq_b_dy is not None:
                groups['processing']['geocode'][
                    'output_posting']['B']['y_posting'] = float(freq_b_dy)

            if self.snap_y is not None:
                groups['processing']['geocode']['y_snap'] = float(self.snap_y)
            if self.snap_x is not None:
                groups['processing']['geocode']['x_snap'] = float(self.snap_x)
            groups['processing']['geocode']['top_left']['x_abs'] = float(x0)
            groups['processing']['geocode']['top_left']['y_abs'] = float(y0)
            groups['processing']['geocode']['bottom_right']['x_abs'] = \
                float(xf)
            groups['processing']['geocode']['bottom_right']['y_abs'] = \
                float(yf)

            print('==========================================================')
            parser.dump(cfg, sys.stdout)
            print('==========================================================')

            if self.output_file:

                output_file = self.output_file
            else:
                output_file = plant.get_temporary_file(append=True, ext='yaml')

            with open(output_file, 'w') as output_file_obj:
                parser.dump(cfg, output_file_obj)

            if self.output_file:
                plant.plant_config.output_files.append(self.output_file)
                print(f'## file saved: {self.output_file}')

            try:
                data = yamale.make_data(output_file, parser='ruamel')
            except yamale.YamaleError as e:
                err_str = (f'Yamale unable to load {self.workflow_name}'
                           f' runconfig yaml for validation.')
                raise yamale.YamaleError(err_str) from e

            schema_file = os.path.join(f'{helpers.WORKFLOW_SCRIPTS_DIR}',
                                       'schemas',
                                       default_runconfig_basename)
            schema = yamale.make_schema(schema_file, parser='ruamel')
            print('verifying output runconfig against schema')
            with plant.PlantIndent():
                print('[OK] runconfig is valid for the'
                      f' {self.workflow_name.upper()} workflow!')

            try:
                yamale.validate(schema, data)
            except yamale.YamaleError as e:
                err_str = (f'Validation fail for {self.workflow_name}'
                           f' runconfig yaml.')
                raise yamale.YamaleError(err_str) from e

            if self.output_file:
                self.output_files.append(self.output_file)
                return self.output_file

    def get_pixel_spacing(self, freq_a_dx, freq_a_dy, freq_b_dx, freq_b_dy,
                          slc_obj):

        if self.snap_y and self.snap_y == 100:
            default_spacing_5_mhz = 100
        else:
            default_spacing_5_mhz = 80

        if 'A' in slc_obj.frequencies:
            freq_a_bandwidth_mhz = int(np.round(
                slc_obj.getSwathMetadata('A').processed_range_bandwidth / 1e6))

            if freq_a_bandwidth_mhz == 5:
                print('## frequency A: 5 MHz mode')
                if not plant.isvalid(freq_a_dx):
                    freq_a_dx = default_spacing_5_mhz
                if not plant.isvalid(freq_a_dy):
                    freq_a_dy = default_spacing_5_mhz
            elif freq_a_bandwidth_mhz == 20:
                print('## frequency A: 20 MHz mode')
                if not plant.isvalid(freq_a_dx):
                    freq_a_dx = 20
                if not plant.isvalid(freq_a_dy):
                    freq_a_dy = 20
            elif freq_a_bandwidth_mhz == 40:
                print('## frequency A: 40 MHz mode')
                if not plant.isvalid(freq_a_dx):
                    freq_a_dx = 10
                if not plant.isvalid(freq_a_dy):
                    freq_a_dy = 10
            elif freq_a_bandwidth_mhz == 77 or freq_a_bandwidth_mhz == 80:
                print('## frequency A: 77 MHz mode')
                if not plant.isvalid(freq_a_dx):
                    freq_a_dx = 20
                if not plant.isvalid(freq_a_dy):
                    freq_a_dy = 20
            else:
                print('WARNING invalid NISAR range bandwidth mode:'
                      f' {freq_a_bandwidth_mhz}')
                if not plant.isvalid(freq_a_dx):
                    freq_a_dx = 20
                if not plant.isvalid(freq_a_dy):
                    freq_a_dy = 20

        if 'B' in slc_obj.frequencies:
            freq_b_bandwidth_mhz = int(np.round(
                slc_obj.getSwathMetadata('B').processed_range_bandwidth / 1e6))

            # if self.workflow_name.upper() == 'GCOV':
            #     freq_b_dx = 100
            #     freq_b_dy = 100
            # else:
            if not plant.isvalid(freq_b_dx):
                freq_b_dx = default_spacing_5_mhz
            if not plant.isvalid(freq_b_dy):
                freq_b_dy = default_spacing_5_mhz

            if freq_b_bandwidth_mhz != 5:
                print('WARNING invalid NISAR range bandwidth mode:'
                      f' {freq_b_bandwidth_mhz}')
            else:
                print('## frequency B: 5 MHz mode')

        return freq_a_dx, freq_a_dy, freq_b_dx, freq_b_dy

    def print_runconfig(self, x0, y0, xf, yf, freq_a_dx, freq_a_dy, freq_b_dx,
                        freq_b_dy, **kwargs):

        workflow_name_upper = self.workflow_name.upper()
        workflow_name = self.workflow_name.lower()

        print('runconfig:', **kwargs)
        print(f'    name: NISAR_L2-L-{workflow_name_upper}_RUNCONFIG',
              **kwargs)
        print('    groups:', **kwargs)
        print('        pge_name_group:', **kwargs)
        print(f'            pge_name: {workflow_name_upper}_L_PGE', **kwargs)
        print('        input_file_group:', **kwargs)
        print('            input_file_path:', self.input_file, **kwargs)
        # print(f'            qa_input_file: output_{workflow_name}/'
        #       f'{workflow_name}.h5', **kwargs)
        print('        dynamic_ancillary_file_group:', **kwargs)
        print('            dem_file:', self.dem_file, **kwargs)
        if self.tec_file:
            print('            tec_file:', self.tec_file, **kwargs)
        if self.external_orbit_file:
            print('            orbit_file:', self.external_orbit_file,
                  **kwargs)
        print('        product_path_group:', **kwargs)
        print(f'            product_path: output_{workflow_name}', **kwargs)
        print(f'            scratch_path: scratch_{workflow_name}', **kwargs)
        print(f'            sas_output_file: {self.sas_output_file}', **kwargs)
        # print('            qa_output_dir: qa_{workflow_name}', **kwargs)
        print('        debug_level_group:', **kwargs)
        print('            debug_switch: false', **kwargs)
        print('        primary_executable:', **kwargs)
        print(f'            product_type: {workflow_name_upper}', **kwargs)
        print('        processing:', **kwargs)
        print('            geocode:', **kwargs)

        if workflow_name_upper == 'GCOV':
            print('                apply_shadow_masking: False', **kwargs)

        print('                output_epsg:', self.epsg, **kwargs)
        print('                output_posting:', **kwargs)
        print('                    A:', **kwargs)
        if freq_a_dx is not None:
            print('                        x_posting:', freq_a_dx, **kwargs)
        else:
            print('                        x_posting:', **kwargs)
        if freq_a_dy is not None:
            print('                        y_posting:', freq_a_dy, **kwargs)
        else:
            print('                        y_posting:', **kwargs)
        print('                    B:', **kwargs)
        if freq_b_dx is not None:
            print('                        x_posting:', freq_b_dx, **kwargs)
        else:
            print('                        x_posting:', **kwargs)
        if freq_b_dy is not None:
            print('                        y_posting:', freq_b_dy, **kwargs)
        else:
            print('                        y_posting:', **kwargs)

        if self.snap_y is not None:
            print('                y_snap:', self.snap_y, **kwargs)
        if self.snap_x is not None:
            print('                x_snap:', self.snap_x, **kwargs)
        print('                top_left:', **kwargs)
        print('                    y_abs:', y0, **kwargs)
        print('                    x_abs:', x0, **kwargs)
        print('                bottom_right:', **kwargs)
        print('                    y_abs:', yf, **kwargs)
        print('                    x_abs:', xf, **kwargs)
        if workflow_name_upper == 'GSLC':
            print('            blocksize:', **kwargs)
            print('                y: 512', **kwargs)
            print('                x: 512', **kwargs)
            print('            flatten: True', **kwargs)

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

        return radar_grid

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

    def _get_coordinates_from_h5_file(self, input_file):
        import shapely.wkt
        nisar_product_obj = open_product(input_file)
        polygon = nisar_product_obj.identification.boundingPolygon
        bounds = shapely.wkt.loads(polygon).bounds
        lat_arr = [bounds[1], bounds[3]]
        lon_arr = [bounds[2], bounds[0]]

        if self.epsg is None:
            zones_list = []
            for i in range(2):
                for j in range(2):
                    zones_list.append(point2epsg(lon_arr[i], lat_arr[j]))
            vals, counts = np.unique(zones_list, return_counts=True)
            self.epsg = int(vals[np.argmax(counts)])
            print('Closest UTM zone: EPSG', self.epsg)


def snap_coord(val, snap, offset, round_func):
    snapped_value = round_func(float(val - offset) / snap) * snap + offset
    return snapped_value


def point2epsg(lon, lat):
    """
     Return EPSG code base on a point
     latitude/longitude coordinates
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(
        'Could not determine projection for {0},{1}'.format(lat, lon))


def lat_lon_to_projected(north, east, epsg):
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
        self_obj = PlantIsce3Runconfig(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
