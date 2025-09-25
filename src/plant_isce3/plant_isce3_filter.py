#!/usr/bin/env python3

import os
import sys
import plant
import plant_isce3

import numpy as np
import isce3

from nisar.products.readers import open_product


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

                            output_file=1)

    plant_isce3.add_arguments(parser,
                              frequency=1)

    parser_metadata = parser.add_mutually_exclusive_group()
    parser_metadata.add_argument('--save-metadata',
                                 dest='flag_save_metadata',
                                 default=None,
                                 action='store_true',
                                 help='Save metadata')

    parser_metadata.add_argument('--do-not-save-metadata',
                                 default=None,
                                 dest='flag_save_metadata',
                                 action='store_false',
                                 help='Do not save metadata')

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

    parser.add_argument('--separate-pol',
                        '--sep-pol',
                        '--separate-pols',
                        '--sep-pols',
                        '--separate-polarizations',
                        '--sep-polarizations',
                        dest='separate_pol',
                        action='store_true',
                        help='Handle polarimetric channels individually,'
                        ' with one output file for each'
                        ' available polarization. Requires the output'
                        ' directory argument: "--output-dir" or "--od"')

    parser.add_argument('--separate-freq',
                        '--sep-freq',
                        '--separate-freqs',
                        '--sep-freqs',
                        '--separate-frequencies',
                        '--sep-frequencies',
                        dest='separate_freq',
                        action='store_true',
                        help='Handle frequencies individually,'
                        ' with one output file for each'
                        ' available frequencies. Requires the output'
                        ' directory argument: "--output-dir" or "--od"')

    return parser


class PlantIsce3Filter(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        self.replace_null = False
        super().__init__(parser, argv)

    def run(self):

        if not self.output_ext:
            self.output_ext = '.tif'

        if self.separate_pol or self.separate_freq:

            if not self.output_dir:
                self.output_dir = '.'

            plant_product_obj = self.load_product()
            if (plant_product_obj.sensor_name != 'NISAR'):
                self.print('ERROR the options --separate-pol and'
                           ' --separate-freq are only available for'
                           ' NISAR products')
                return
            freq_pols = plant_product_obj.nisar_product_obj.polarizations

            frequency_orig = self.frequency
            band_orig = self.band

            output_file_orig = self.output_file

            if self.separate_freq:
                freqs_iterator = freq_pols.items()
            else:
                frequency = plant_product_obj.get_frequency_str()
                pols_iterator = freq_pols[frequency]
                freqs_iterator = [[frequency, pols_iterator]]

            if output_file_orig:
                self.print(f'## output file template: {output_file_orig}')

            ret_list = []
            for freq, pols in freqs_iterator:
                if (frequency_orig is not None and
                        frequency_orig != freq):
                    continue

                if not self.separate_pol:

                    if output_file_orig:
                        self.output_file = output_file_orig
                        self.output_file = self.output_file.replace(
                            '{frequency}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{freq}', self.frequency)
                    else:
                        self.output_file = os.path.join(
                            self.output_dir,
                            f'data_freq_{freq}{self.output_ext}')

                    self.frequency = freq

                    if (self.output_skip_if_existent and
                            plant.isfile(self.output_file)):
                        print('INFO output file %s already exist, '
                              'skipping execution..' % self.output_file)
                        continue

                    ret = self.run_filter()
                    continue

                for band, pol in enumerate(pols):
                    if (band_orig is not None and
                            band_orig != band):
                        continue

                    if band_orig is not None and band != band_orig:
                        continue

                    self.frequency = freq
                    self.band = band

                    if output_file_orig:
                        self.output_file = output_file_orig
                        self.output_file = self.output_file.replace(
                            '{frequency}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{freq}', self.frequency)
                        self.output_file = self.output_file.replace(
                            '{polarization}', pol)
                        self.output_file = self.output_file.replace(
                            '{pol}', pol)
                    else:
                        self.output_file = os.path.join(
                            self.output_dir,
                            f'data_freq_{freq}_{pol}{self.output_ext}')

                    if (self.output_skip_if_existent and
                            plant.isfile(self.output_file)):
                        print('INFO output file %s already exist, '
                              'skipping execution..' % self.output_file)
                        continue
                    ret = self.run_filter()

                ret_list.append(ret)
            return ret_list

        elif (not self.output_file and not self.output_dir and
                not self.output_ext):
            self.parser.print_usage()
            self.print('ERROR one the following argument is required:'
                       ' --output-file, --output-dir, --output-ext')
            sys.exit(1)
        elif (not self.output_file and
              (self.output_dir or
               self.output_ext)):
            self.parser.print_usage()
            self.print('ERROR this script only accepts --output-dir or'
                       ' --output-ext in --separate-pol or'
                       ' --separate-freq modes')
            sys.exit(1)
        return self.run_filter()

    def run_filter(self):
        self.print(f'input file: {str(self.input_file)}')

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        nlooks_y = self.nlooks_y
        nlooks_x = self.nlooks_x
        transform_square = self.transform_square
        plant_transform_obj = self.plant_transform_obj

        self.nlooks_y = 1
        self.nlooks_x = 1
        self.transform_square = None
        self.plant_transform_obj = None

        plant_product_obj = self.load_product()
        input_raster = self.get_input_raster_from_nisar_slc(
            plant_product_obj=plant_product_obj)

        self.nlooks_y = nlooks_y
        self.nlooks_x = nlooks_x
        self.transform_square = transform_square
        self.plant_transform_obj = plant_transform_obj

        if self.flag_save_metadata is not False:
            metadata_dict = {}
            metadata_dict['INPUT_FILE'] = self.input_file
            if self.frequency is not None:
                metadata_dict['FREQUENCY'] = self.frequency
            if self.nlooks_y is not None:
                metadata_dict['NLOOKS_Y'] = nlooks_y
            if self.nlooks_x is not None:
                metadata_dict['NLOOKS_X'] = nlooks_x
            if self.output_format is not None:
                metadata_dict['OUTPUT_FORMAT'] = self.output_format
        else:
            metadata_dict = None

        plant_isce3.multilook_isce3(input_raster, self.output_file,
                                    self.nlooks_y, self.nlooks_x,
                                    transform_square=self.transform_square,
                                    output_format=self.output_format,
                                    metadata_dict=metadata_dict,
                                    block_nlines=self.block_nlines)

        ret_dict = {}
        ret_dict['output_file'] = self.output_file
        plant.append_output_file(self.output_file)
        self.update_output_format(ret_dict)

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
