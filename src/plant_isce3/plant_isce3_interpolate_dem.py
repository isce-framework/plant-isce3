#!/usr/bin/env python3

import plant
import plant_isce3
from osgeo import gdal
import isce3
import numpy as np


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            dem_file=2,
                            default_options=1,
                            output_file=2,
                            geo=1)

    parser.add_argument('--epsg',
                        action='store',
                        dest='epsg',
                        type=int,
                        default=None,
                        help='EPSG code for output grid')

    parser.add_argument('--dem-interp-method',
                        dest='dem_interp_method',
                        type=str,
                        help='DEM interpolation method. Options:'
                        ' sinc, bilinear, bicubic, nearest, biquintic')

    return parser


class PlantIsce3InterpolateDem(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        dem_raster = isce3.io.Raster(self.dem_file)
        if self.epsg is None:
            self.epsg = dem_raster.get_epsg()

        geogrid_obj = isce3.product.GeoGridParameters(
            start_x=self.plant_geogrid_obj.x0,
            start_y=self.plant_geogrid_obj.y0,
            spacing_x=self.plant_geogrid_obj.step_x,
            spacing_y=self.plant_geogrid_obj.step_y,
            width=int(self.plant_geogrid_obj.width),
            length=int(self.plant_geogrid_obj.length),
            epsg=self.epsg)

        geogrid_obj.print()

        nbands = 1
        shape = [nbands, geogrid_obj.length, geogrid_obj.width]

        interpolated_dem_raster = self._get_raster(
            self.output_file, nbands=shape[0], length=shape[1],
            width=shape[2])
        output_obj_list = [interpolated_dem_raster]

        dem_interp_method = _get_dem_interp_method(self.dem_interp_method)

        isce3.geogrid.relocate_raster(dem_raster,
                                      geogrid_obj,
                                      interpolated_dem_raster,
                                      dem_interp_method)

        for obj in output_obj_list:
            del obj

        for f in plant.plant_config.output_files:
            self.print(f'## file saved: {f}')


def _get_dem_interp_method(dem_interp_method):
    if (dem_interp_method is None or
            dem_interp_method.upper() == 'BIQUINTIC'):
        return isce3.core.DataInterpMethod.BIQUINTIC
    if (dem_interp_method.upper() == 'SINC'):
        return isce3.core.DataInterpMethod.SINC
    if (dem_interp_method.upper() == 'BILINEAR'):
        return isce3.core.DataInterpMethod.BILINEAR
    if (dem_interp_method.upper() == 'BICUBIC'):
        return isce3.core.DataInterpMethod.BICUBIC
    if (dem_interp_method.upper() == 'NEAREST'):
        return isce3.core.DataInterpMethod.NEAREST
    raise NotImplementedError


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3InterpolateDem(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
