#!/usr/bin/env python3

import os
import glob
import plant
import plant_isce3
import isce3
from osgeo import gdal
import numpy as np
from plant_isce3.readers import SLC


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=2,
                            default_options=1,
                            multilook=1,
                            output_dir=2)

    plant_isce3.add_arguments(parser,
                              burst_ids=1,
                              frequency=1,
                              epsg=1,
                              native_doppler_grid=1,
                              orbit_files=1)

    parser.add_argument('--rdr2geo-threshold',
                        type=float,
                        dest='threshold_rdr2geo',
                        help='Range convergence threshold for rdr2geo')

    parser.add_argument('--rdr2geo-num-iter',
                        '--rdr2geo-numiter',
                        type=float,
                        dest='numiter_rdr2geo',
                        help='Maximum number of iterations for rdr2geo')

    parser.add_argument('--rdr2geo-extra-iter',
                        '--rdr2geo-extraiter',
                        type=float,
                        dest='extraiter_rdr2geo',
                        help='Extra iterations for rdr2geo')

    parser.add_argument('--lines-per-block',
                        type=int,
                        dest='lines_per_block',
                        help='Lines per block')

    parser.add_argument('--out-x',
                        action='store_true',
                        dest='flag_x',
                        help='Save X coordinates')

    parser.add_argument('--out-y',
                        action='store_true',
                        dest='flag_y',
                        help='Save Y coordinates')

    parser.add_argument('--out-z',
                        action='store_true',
                        dest='flag_z',
                        help='Save DEM elevation')

    parser.add_argument('--out-incidence-angle',
                        '--out-inc-angle',
                        action='store_true',
                        dest='flag_incidence_angle',
                        help='Save the incidence angle')

    parser.add_argument('--out-heading',
                        action='store_true',
                        dest='flag_heading_angle',
                        help='Save the heading angle')

    parser.add_argument('--out-local-incidence-angle',
                        '--out-local-inc-angle',
                        action='store_true',
                        dest='flag_local_incidence_angle',
                        help='Save the local incidence angle')

    parser.add_argument('--out-projection-angle',
                        '--out-psi-angle',
                        '--out-local-projection-angle',
                        '--out-local-psi-angle',
                        action='store_true',
                        dest='flag_projection_angle',
                        help='Save the projection angle')

    parser.add_argument('--out-simulated-amplitude',
                        '--out-simulated-amp',
                        '--out-sim-amp',
                        action='store_true',
                        dest='flag_simulated_amplitude',
                        help='Save the simulated amplitude')

    parser.add_argument('--out-layover-shadow-mask',
                        action='store_true',
                        dest='flag_layover_shadow_mask',
                        help='Save the layover/shadow mask')

    parser.add_argument('--out-los',
                        action='store_true',
                        dest='flag_los',
                        help='Save line-of-sight unit vector')

    return parser


class PlantIsce3Topo(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        plant_product_obj = self.load_product()
        radar_grid_ml = plant_product_obj.get_radar_grid_ml()
        orbit = plant_product_obj.get_orbit()
        doppler = plant_product_obj.get_grid_doppler()

        dem_raster = plant_isce3.get_isce3_raster(self.dem_file)
        if self.epsg is None:
            self.epsg = dem_raster.get_epsg()

        print(f'output EPSG: {self.epsg}')

        print('Radar grid:')
        print('    length:', radar_grid_ml.length)
        print('    width:', radar_grid_ml.width)

        ellipsoid = isce3.core.Ellipsoid()

        topo_kwargs = {}
        if self.threshold_rdr2geo is not None:
            topo_kwargs['threshold'] = self.threshold_rdr2geo
        if self.numiter_rdr2geo is not None:
            topo_kwargs['numiter'] = self.numiter_rdr2geo
        if self.extraiter_rdr2geo is not None:
            topo_kwargs['extraiter'] = self.extraiter_rdr2geo
        if self.lines_per_block is not None:
            topo_kwargs['lines_per_block'] = self.lines_per_block

        topo = isce3.geometry.Rdr2Geo(radar_grid_ml,
                                      orbit,
                                      ellipsoid,
                                      epsg_out=self.epsg,
                                      doppler=doppler,
                                      **topo_kwargs)

        flag_all = (self.flag_x is not True and
                    self.flag_y is not True and
                    self.flag_z is not True and
                    self.flag_incidence_angle is not True and
                    self.flag_heading_angle is not True and
                    self.flag_local_incidence_angle is not True and
                    self.flag_projection_angle is not True and
                    self.flag_simulated_amplitude is not True and
                    self.flag_layover_shadow_mask is not True and
                    self.flag_los is not True)

        if self.output_dir and not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if flag_all:
            print('*** create all layers')
            topo.topo(dem_raster, self.output_dir)

        else:
            output_obj_list = []

            nbands = 1

            shape = [nbands, radar_grid_ml.length, radar_grid_ml.width]

            print('*** create some layers with shape:', shape)

            x_raster = self._get_raster(
                self.output_dir, 'x', np.float32, shape,
                output_obj_list,
                self.flag_x)
            y_raster = self._get_raster(
                self.output_dir, 'y', np.float64, shape, output_obj_list,
                self.flag_y)
            height_raster = self._get_raster(
                self.output_dir, 'z', np.float64, shape,
                output_obj_list,
                self.flag_z)
            incidence_angle_raster = self._get_raster(
                self.output_dir, 'inc', np.float32, shape,
                output_obj_list,
                self.flag_incidence_angle)
            heading_angle_raster = self._get_raster(
                self.output_dir, 'hdg', np.float32, shape,
                output_obj_list,
                self.flag_heading_angle)
            local_incidence_angle_raster = self._get_raster(
                self.output_dir, 'localInc', np.float32, shape,
                output_obj_list,
                self.flag_local_incidence_angle)
            local_projection_angle_raster = self._get_raster(
                self.output_dir, 'localPsi', np.float32, shape,
                output_obj_list,
                self.flag_projection_angle)
            simulated_amplitude_raster = self._get_raster(
                self.output_dir, 'simamp', np.float32, shape,
                output_obj_list,
                self.flag_projection_angle)
            layover_shadow_raster = self._get_raster(
                self.output_dir, 'layoverShadowMask', np.float32, shape,
                output_obj_list,
                self.flag_layover_shadow_mask)
            los_east_raster = self._get_raster(
                self.output_dir, 'los_east', np.float32, shape,
                output_obj_list,
                self.flag_los)
            los_north_raster = self._get_raster(
                self.output_dir, 'los_north', np.float32, shape,
                output_obj_list,
                self.flag_los)

            topo.topo(
                dem_raster=dem_raster,
                x_raster=x_raster,
                y_raster=y_raster,
                height_raster=height_raster,
                incidence_angle_raster=incidence_angle_raster,
                heading_angle_raster=heading_angle_raster,
                local_incidence_angle_raster=local_incidence_angle_raster,
                local_psi_raster=local_projection_angle_raster,
                simulated_amplitude_raster=simulated_amplitude_raster,
                layover_shadow_raster=layover_shadow_raster,
                ground_to_sat_east_raster=los_east_raster,
                ground_to_sat_north_raster=los_north_raster)

        if self.output_dir and not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        topo.topo(dem_raster, self.output_dir)

        if flag_all:
            output_obj_list = glob.glob(os.path.join(self.output_dir,
                                                     '*.rdr'))
        for output_file in output_obj_list:
            plant.append_output_file(output_file)

    def _get_raster(self, output_dir, ds_name, dtype, shape,
                    output_obj_list, flag_save_vector):
        if flag_save_vector is not True:
            return None

        output_file = os.path.join(output_dir, ds_name) + '.tif'
        print(f'*** {ds_name}', output_file)
        raster_obj = plant_isce3.get_isce3_raster(
            output_file,
            shape[2],
            shape[1],
            shape[0],
            gdal.GDT_Float32,
            "GTiff")
        plant.append_output_file(output_file)
        output_obj_list.append(raster_obj)
        return raster_obj


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Topo(parser, argv)
        ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
