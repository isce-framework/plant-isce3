#!/usr/bin/env python3

import os
import glob
import plant
import isce3
from nisar.products.readers import SLC


def get_parser():
    '''
    Command line parser.
    '''
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

    parser.add_argument('--native-doppler',
                        dest='native_doppler',
                        default=False,
                        action='store_true',
                        help='Native Doppler.')

    return parser


class PlantISCE3Topo(plant.PlantScript):

    def __init__(self, parser, argv=None):
        '''
        class initialization
        '''
        super().__init__(parser, argv)

    def run(self):
        '''
        run main method
        '''
        print('*** 0')
        if self.input_key and self.input_key == 'B':
            frequency_str = 'B'
        else:
            frequency_str = 'A'

        slc_obj = SLC(hdf5file=self.input_file)
        orbit = slc_obj.getOrbit()
        doppler = self._get_doppler(slc_obj)

        dem_raster = isce3.io.Raster(self.dem_file)
        if self.epsg is None:
            self.epsg = dem_raster.get_epsg()


        if self.nlooks_az is None:
            self.nlooks_az = 1
        if self.nlooks_rg is None:
            self.nlooks_rg = 1

        print(f'output EPSG: {self.epsg}')

        radar_grid_ml = self._get_radar_grid(slc_obj,
                                             frequency_str)

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


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantISCE3Topo(parser, argv)
        ret = self_obj.run()
        return ret

if __name__ == '__main__':
    main()
