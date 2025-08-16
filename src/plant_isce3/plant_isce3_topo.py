#!/usr/bin/env python3

import os
import glob
import plant
import plant_isce3
import isce3
from nisar.products.readers import SLC


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

        if self.output_dir and not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        topo.topo(dem_raster, self.output_dir)
        output_filelist = glob.glob(os.path.join(self.output_dir,
                                                 '*.rdr'))
        for output_file in output_filelist:
            plant.append_output_file(output_file)


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Topo(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
