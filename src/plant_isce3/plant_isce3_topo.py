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

    parser.add_argument('--epsg',
                        dest='epsg',
                        type=int,
                        help='EPSG code for output grids.')

    parser.add_argument('--native-doppler-grid',
                        dest='native_doppler_grid',
                        default=False,
                        action='store_true',
                        help='Consider native Doppler grid (skewed geometry)')

    return parser


class PlantIsce3Topo(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        slc_obj = SLC(hdf5file=self.input_file)
        frequency_str = list(slc_obj.polarizations.keys())[0]

        orbit = slc_obj.getOrbit()
        doppler = self.get_doppler_grid_lut(slc_obj)

        dem_raster = isce3.io.Raster(self.dem_file)
        if self.epsg is None:
            self.epsg = dem_raster.get_epsg()

        if self.nlooks_az is None:
            self.nlooks_az = 1
        if self.nlooks_rg is None:
            self.nlooks_rg = 1

        print(f'output EPSG: {self.epsg}')

        radar_grid_ml = self.get_radar_grid(slc_obj,
                                            frequency_str)

        print('radar grid:')
        print('    length:', radar_grid_ml.length)
        print('    width:', radar_grid_ml.width)

        ellipsoid = isce3.core.Ellipsoid()

        topo = isce3.geometry.Rdr2Geo(radar_grid_ml,
                                      orbit,
                                      ellipsoid,
                                      epsg_out=self.epsg,
                                      doppler=doppler)

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
