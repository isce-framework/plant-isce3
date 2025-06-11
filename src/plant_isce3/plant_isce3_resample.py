#!/usr/bin/env python3

import plant
import plant_isce3

import isce3


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=2,

                            default_options=1,
                            output_file=2,
                            geo=1)

    plant_isce3.add_arguments(parser,
                              data_interp_method=1,
                              epsg=1)

    return parser


class PlantIsce3Resample(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        dem_raster = isce3.io.Raster(self.input_file)
        if self.epsg is None:
            self.epsg = dem_raster.get_epsg()

        geogrid_obj = isce3.product.GeoGridParameters(
            start_x=self.plant_geogrid_obj.x0,
            start_y=self.plant_geogrid_obj.y0,
            spacing_x=self.plant_geogrid_obj.step_x,
            spacing_y=self.plant_geogrid_obj.step_y,
            width=int(self.plant_geogrid_obj.width),
            length=int(self.plant_geogrid_obj.length),
            epsg=self.plant_geogrid_obj.epsg)

        geogrid_obj.print()

        nbands = 1
        shape = [nbands, geogrid_obj.length, geogrid_obj.width]

        output_raster = self._create_output_raster(
            self.output_file, nbands=shape[0], length=shape[1],
            width=shape[2])
        output_obj_list = [output_raster]

        if not self.data_interp_method:
            self.data_interp_method = 'bicubic'

        data_interp_method = _get_data_interp_method(
            self.data_interp_method)

        isce3.geogrid.relocate_raster(dem_raster,
                                      geogrid_obj,
                                      output_raster,
                                      data_interp_method)

        plant.plant_config.output_files.append(self.output_file)
        self.print(f'## file saved: {self.output_file}')

        for obj in output_obj_list:
            del obj


def _get_data_interp_method(data_interp_method):
    if (data_interp_method is None or
            data_interp_method.upper() == 'BIQUINTIC'):
        return isce3.core.DataInterpMethod.BIQUINTIC
    if (data_interp_method.upper() == 'SINC'):
        return isce3.core.DataInterpMethod.SINC
    if (data_interp_method.upper() == 'BILINEAR'):
        return isce3.core.DataInterpMethod.BILINEAR
    if (data_interp_method.upper() == 'BICUBIC'):
        return isce3.core.DataInterpMethod.BICUBIC
    if (data_interp_method.upper() == 'NEAREST'):
        return isce3.core.DataInterpMethod.NEAREST
    raise NotImplementedError


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Resample(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
