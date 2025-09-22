#!/usr/bin/env python3

import os
import sys
import plant
import plant_isce3

import numpy as np
import isce3
from osgeo import gdal


def get_parser():

    descr = ''
    epilog = ''

    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=2,
                            band=1,
                            default_output_options=1,
                            default_flags=1,
                            output_format=1,
                            multilook=1,
                            output_dir=1,
                            separate=1,
                            output_file=1)

    plant_isce3.add_arguments(parser,
                              frequency=1)

    parser_matrix = parser.add_mutually_exclusive_group()
    parser_matrix.add_argument('--cov',
                               '--cov-matrix',
                               '--covariance-matrix',
                               dest='flag_covariance_matrix',
                               action='store_true',
                               help='Generate covariance matrix [C].')

    parser_matrix.add_argument('--coh',
                               '--coh-matrix',
                               '--coherency-matrix',
                               dest='flag_coherency_matrix',
                               action='store_true',
                               help='Generate coherency matrix [T].')

    parser.add_argument('--square', '--sq',
                        action='store_true',
                        dest='transform_square',
                        help='Square of input')

    parser.add_argument('--nlines-block',
                        '--block-nlines',
                        dest='block_nlines',
                        type=int,
                        default=4096,
                        help='Number of lines per block')

    return parser


class PlantIsce3Filter(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        self.replace_null = False
        super().__init__(parser, argv)

    def run(self):

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        if self.separate and self.flag_covariance_matrix:
            self.print('ERROR separate and covariance matrix modes'
                       ' cannot be used together')
            return

        if not self.output_ext:
            self.output_ext = '.bin'

        if (not self.output_file and not self.output_dir and
                not self.output_ext):
            self.parser.print_usage()
            self.print('ERROR one the following argument is required: '
                       '--output-file, --output-dir, --output-ext')
            sys.exit(1)
        elif (not self.output_file and
              (self.output_dir or
               self.output_ext)):
            self.parser.print_usage()
            self.print('ERROR this script only accepts --output-dir or '
                       '--output-ext in --separate mode')
            sys.exit(1)
        return self.run_filter()

    def run_filter(self):
        self.print(f'input file: {str(self.input_file)}')

        nlooks_az = self.nlooks_az
        nlooks_rg = self.nlooks_rg
        transform_square = self.transform_square
        plant_transform_obj = self.plant_transform_obj

        self.nlooks_az = 1
        self.nlooks_rg = 1
        self.transform_square = None
        self.plant_transform_obj = None

        plant_product_obj = self.load_product()
        input_raster = self.get_input_raster_from_nisar_slc(

            plant_product_obj=plant_product_obj)

        self.nlooks_az = nlooks_az
        self.nlooks_rg = nlooks_rg
        self.transform_square = transform_square
        self.plant_transform_obj = plant_transform_obj

        plant_isce3.multilook_isce3(input_raster, self.output_file,
                                    self.nlooks_az, self.nlooks_rg,
                                    transform_square=self.transform_square,
                                    block_nlines=self.block_nlines)

        ret_dict = {}
        ret_dict['output_file'] = self.output_file
        plant.append_output_file(self.output_file)
        self.update_output_format(ret_dict)

        return self.output_file

    def _filter(self, input_raster_file, output_file):

        input_raster = isce3.io.Raster(input_raster_file)

        width_ml = input_raster.width // self.nlooks_rg
        length_ml = input_raster.length // self.nlooks_az

        exponent = 2 if self.transform_square else 0

        if exponent % 2 == 0:
            output_dtype = gdal.GDT_Float32
        else:
            output_dtype = gdal.GDT_CFloat32
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        nbands = input_raster.num_bands
        output_raster = isce3.io.Raster(output_file,

                                        width_ml,
                                        length_ml,
                                        nbands,
                                        output_dtype,
                                        "ENVI")

        if self.block_nlines is not None and plant.isvalid(self.block_nlines):
            block_nlines = ((self.block_nlines // self.nlooks_az) *
                            self.nlooks_az)

        print('block number of lines:', block_nlines)
        n_blocks = int(np.ceil(float(input_raster.length) / block_nlines))

        for band in range(nbands):

            with plant.PrintProgress(n_blocks) as progress_obj:

                for block in range(n_blocks):
                    progress_obj.print_progress(block)

                    start_line = block_nlines * block
                    end_line = min([block_nlines * (block + 1),
                                    input_raster.length + 1])
                    block_array = input_raster.get_block(
                        key=np.s_[start_line:end_line, :],
                        band=band + 1)

                    if exponent > 1:
                        block_array = np.absolute(block_array) ** 2

                    multilooked_image = isce3.signal.multilook_nodata(
                        block_array,
                        self.nlooks_az, self.nlooks_rg, np.nan)

                    start_line_ml = start_line // self.nlooks_az
                    end_line_ml = end_line // self.nlooks_az
                    output_raster.set_block(
                        key=np.s_[start_line_ml:end_line_ml, :],
                        value=multilooked_image,
                        band=band + 1)

        del input_raster
        del output_raster

        input_gdal_ds = gdal.Open(input_raster_file)
        geotransform = input_gdal_ds.GetGeoTransform()
        if geotransform is not None:
            geotransform = list(geotransform)
            projection = input_gdal_ds.GetProjection()
            input_gdal_ds.FlushCache()
            input_gdal_ds.Close()

            del input_gdal_ds

            print('geotransform (original):', geotransform)
            geotransform[1] = geotransform[1] * self.nlooks_rg
            geotransform[5] = geotransform[5] * self.nlooks_az

            print('geotransform (multilooked):', geotransform)
            print('projection:', projection)

            output_gdal_ds = gdal.Open(output_file, gdal.GA_Update)
            output_gdal_ds.SetGeoTransform(geotransform)
            output_gdal_ds.SetProjection(projection)
            output_gdal_ds.FlushCache()
            output_gdal_ds.Close()

            del output_gdal_ds

    def get_filter_kwargs(self):

        kwargs = {}
        kwargs['nlooks'] = self.nlooks
        kwargs['verbose'] = self.verbose

        return kwargs


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Filter(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
