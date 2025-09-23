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
                                    output_format=self.output_format,
                                    block_nlines=self.block_nlines)

        ret_dict = {}
        ret_dict['output_file'] = self.output_file
        plant.append_output_file(self.output_file)
        plant_isce3.update_output_format(ret_dict)

        return self.output_file

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
