#!/usr/bin/env python3

import os
import sys
import plant
import plant_isce3
import numpy as np
from plant_isce3.readers import open_product


def get_parser():

    descr = ''
    epilog = ''

    parser = plant.argparse(epilog=epilog,
                            description=descr,

                            input_file=2,

                            output_skip_if_existent=1,
                            default_flags=1,
                            multilook=1,
                            output_dir=2

                            )

    plant_isce3.add_arguments(parser,
                              nlooks_by_frequency=1,
                              frequency=1)

    parser.add_argument('--plot-date',
                        type=str,
                        dest='plot_date',
                        help='Date string to include in plot titles')

    parser.add_argument('--plot-dataset-name',
                        type=str,
                        dest='plot_dataset_name',
                        help='Dataset name to include in plot titles')

    parser.add_argument('--profile-filter-size',
                        dest='profile_filter_size',
                        type=int,
                        help='Profile filter size. "-1" to disable filtering.')

    parser.add_argument('--generate-elevation-profiles',
                        action='store_true',
                        dest='flag_generate_elevation_profiles',
                        help='Generate elevation profiles')

    parser.add_argument('--remove-cross-mul-file',
                        '--remove-cross-multiplication-file',
                        '--remove-cross-mul-files',
                        '--remove-cross-multiplication-files',
                        action='store_true',
                        dest='remove_cross_mul_file',
                        help=('Remove cross multiplication file after'
                              ' processing.'))

    return parser


MAX_N_LINES = 20000


class PlantIsce3OffDiagAnalysis(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        self.replace_null = False
        super().__init__(parser, argv)

    def run(self):

        nisar_product_obj = open_product(self.input_file)
        freq_pol_dict = nisar_product_obj.polarizations
        plant_product_obj = self.load_product()

        for freq, pol_list in freq_pol_dict.items():
            if self.frequency is not None and freq != self.frequency:
                continue

            pol_list_sorted = sorted(pol_list)

            print(f'Processing frequency: {freq}')
            print(f'Available polarizations: {pol_list_sorted}')

            for i, pol_1 in enumerate(pol_list_sorted):
                for j, pol_2 in enumerate(pol_list_sorted):
                    if i >= j:
                        continue
                    self.process_polarimetric_pair(
                        plant_product_obj, nisar_product_obj, freq,
                        pol_1, pol_2)

    def get_metadata(self, freq,

                     nlooks_y, nlooks_x, profile_filter_size):
        metadata_dict = {
            'REFERENCE_RSLC': self.input_file,
            'FREQUENCY': freq,

            'PROFILE_FILTER_SIZE': profile_filter_size,
            'NLOOKS_Y': nlooks_y,
            'NLOOKS_X': nlooks_x
        }

        return metadata_dict

    def process_polarimetric_pair(self, plant_product_obj, nisar_product_obj,
                                  freq, pol_1, pol_2):

        flag_has_input_data_exception = \
            plant_product_obj.get_nisar_identification_scalar(
                'hasInputDataException')

        data_exception_str = \
            f' (data exception: {flag_has_input_data_exception})'

        nlooks_y, nlooks_x = self.get_nlooks(frequency=freq)

        print(f'   Processing polarization pair: {pol_1}, {pol_2}')
        pol_1_ref = (f'HDF5:{self.input_file}://science/LSAR/RSLC/swaths/'
                     f'frequency{freq}/{pol_1}')
        pol_2_ref = (f'HDF5:{self.input_file}://science/LSAR/RSLC/swaths/'
                     f'frequency{freq}/{pol_2}')

        cross_mul_file = os.path.join(
            self.output_dir,
            f'{pol_1}{pol_2}_freq_{freq}.tif')
        multilooked_file = os.path.join(
            self.output_dir,
            f'{pol_1}{pol_2}_freq_{freq}_ml_{nlooks_x}_{nlooks_y}.tif')

        beta0_ml_file = os.path.join(
            self.output_dir,
            f'{pol_1}{pol_1}_freq_{freq}_ml_{nlooks_x}_{nlooks_y}.tif')

        slant_range_file = os.path.join(
            self.output_dir, 'slant_range',
            f'slant_range_{freq}.tif')

        profile_x_file = os.path.join(
            self.output_dir, 'slant_range',
            f'{pol_1}{pol_2}_freq_{freq}_profile_x.tif')

        profile_phase_processed_file = os.path.join(
            self.output_dir, 'slant_range',
            f'{pol_1}{pol_2}_freq_{freq}_profile_x_phase_filtered_no_offset'
            '.tif')

        profile_x_file_png = os.path.join(
            self.output_dir,
            f'{pol_1}{pol_2}_freq_{freq}_profile_x.png')

        profile_x_no_offset_file_png = os.path.join(
            self.output_dir,
            f'{pol_1}{pol_2}_freq_{freq}_profile_x_no_offset.png')

        if (not self.output_skip_if_existent or
                (not os.path.isfile(multilooked_file) and
                 not os.path.isfile(profile_x_file))):

            if (freq == 'A' and (not self.output_skip_if_existent or
                                 not os.path.isfile(cross_mul_file))):

                length = plant.read_image(pol_1_ref).length

                image_list = []
                for count in range(length // MAX_N_LINES + 1):

                    cross_mul_part_file = os.path.join(
                        self.output_dir,
                        f'{pol_1}{pol_2}_freq_{freq}_part_{count}.tif')

                    plant.util(
                        pol_1_ref, pol_2_ref, output_file=cross_mul_part_file,
                        cross_mul=True,
                        force=True,
                        output_skip_if_existent=self.output_skip_if_existent,
                        row=f'{count * MAX_N_LINES}:{(count + 1) * MAX_N_LINES}')

                    image_list.append(cross_mul_part_file)
                plant.util(
                    image_list,
                    output_file=cross_mul_file,
                    force=True,
                    concatenate_y=len(image_list) > 1,
                    output_skip_if_existent=self.output_skip_if_existent)

                if os.path.isfile(cross_mul_file):

                    for count in range(length // MAX_N_LINES + 1):
                        cross_mul_part_file = os.path.join(
                            self.output_dir,
                            f'{pol_1}{pol_2}_freq_{freq}_part_{count}.tif')
                        os.remove(cross_mul_part_file)

            else:
                plant.util(
                    pol_1_ref, pol_2_ref, output_file=cross_mul_file,
                    cross_mul=True, force=True,
                    output_skip_if_existent=self.output_skip_if_existent)

        if (not self.output_skip_if_existent or
                not os.path.isfile(multilooked_file)):
            multilooked_complex = plant.filter(
                cross_mul_file,
                output_file=multilooked_file,
                nlooks_x=nlooks_x,
                nlooks_y=nlooks_y,
                snap_to_multilook_grid=True,
                output_skip_if_existent=self.output_skip_if_existent,
                force=True).image
        else:
            multilooked_complex = plant.read_image(multilooked_file).image

        masked_image = self.get_masked_nisar_data_radar_coordinates(
            nisar_product_obj, multilooked_complex,
            freq, nlooks_y, nlooks_x)

        profile_x = np.nanmean(masked_image, axis=0)

        if self.profile_filter_size is None:
            profile_filter_size = profile_x.size // 10
        else:
            profile_filter_size = self.profile_filter_size

        metadata_dict = self.get_metadata(freq, nlooks_y, nlooks_x,
                                          self.profile_filter_size)

        plant_product_obj = self.load_product()
        radar_grid_ml = plant_product_obj.get_radar_grid_ml(
            frequency=freq)

        if self.plot_date:
            date_str = f' - {self.plot_date}'
        else:
            date_str = ''

        if self.plot_dataset_name:
            title_no_offset = (
                f'{self.plot_dataset_name} - RSLC Phase({pol_1}.{pol_2}*)'
                f' Freq. {freq}' + date_str + data_exception_str +
                ' (offset removed) [rad]')
        else:
            title_no_offset = (
                f'RSLC Phase({pol_1}.{pol_2}*)'
                f' Freq. {freq}' + date_str + data_exception_str +
                ' (offset removed) [rad]')
        if self.flag_generate_elevation_profiles:

            pol_pair_str = f'{pol_1}.conj_{pol_2}'
            elevation_profile, phase_elevation_profile, _ = \
                self.generate_elevation_profiles(
                    self.output_dir,
                    nisar_product_obj, freq,
                    pol_pair_str,
                    masked_image,
                    radar_grid_ml,
                    metadata_dict,
                    profile_filter_size,

                    suffix_ml='',
                    prefix='',
                    flag_phase=True)

            elevation_profile_png_file = os.path.join(
                self.output_dir,
                f'elevation_profile_{pol_pair_str}_freq_{freq}.png')
            plant.display(
                elevation_profile,
                phase_elevation_profile,
                first_input_as_x=True,
                title=title_no_offset,
                marker='',
                dpi=150,
                fontsize=9,
                linewidth=2,

                xlabel='Elevation [deg]',
                output_file=elevation_profile_png_file,
                ymax=3.14,
                ymin=-3.14,
                ylabel='Phase [rad]',
                hline="-0.5,0.5",
                stats_linestyle='dashed',
                stats_linewidth=1,
                stats_linecolor='red',
                no_show=True,
                force=True)

        if not os.path.isfile(slant_range_file):

            slant_ranges_ml = np.array(radar_grid_ml.slant_ranges)
            if masked_image.shape[1] < len(slant_ranges_ml):
                slant_ranges_ml = slant_ranges_ml[:masked_image.shape[1]]

            plant.save_image(slant_ranges_ml,
                             output_file=slant_range_file,
                             metadata=metadata_dict,
                             force=True)
        else:
            slant_ranges_ml = plant.read_image(slant_range_file).image

        slant_ranges_ml_km = slant_ranges_ml / 1000

        if self.plot_dataset_name:
            title = (f'{self.plot_dataset_name} - RSLC Phase({pol_1}.{pol_2}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')
        else:
            title = (f'RSLC Phase({pol_1}.{pol_2}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')

        if profile_filter_size > 0:
            print('applying profile filter of size',
                  profile_filter_size)

            profile_x = plant.filter_data(
                profile_x,
                mean=[1, profile_filter_size])

        profile_x_phase = np.angle(profile_x)

        plant.display(slant_ranges_ml_km,
                      profile_x_phase,
                      first_input_as_x=True,

                      title=title,
                      dpi=150,
                      fontsize=9,
                      ymin=-3.14,
                      ymax=3.14,
                      label_y='Phase [rad]',
                      output_file=profile_x_file_png,
                      label_x='Slant range [km]',
                      no_show=True,

                      force=True)

        profile_x_mean_phase = np.angle(np.nanmean(profile_x))
        profile_no_offset = profile_x * np.exp(-1j * profile_x_mean_phase)
        profile_no_offset_phase = np.angle(profile_no_offset)

        plant.display(slant_ranges_ml_km,
                      profile_no_offset_phase,
                      first_input_as_x=True,
                      title=title_no_offset,
                      dpi=150,
                      fontsize=9,
                      ymin=-3.14,
                      ymax=3.14,
                      hline="-0.5,0.5",
                      stats_linestyle='dashed',
                      stats_linewidth=1,
                      stats_linecolor='red',
                      label_y='Phase [rad]',
                      output_file=profile_x_no_offset_file_png,
                      label_x='Slant range [km]',
                      no_show=True,

                      force=True)

        masked_image_phase = np.angle(masked_image *
                                      np.exp(-1j * profile_x_mean_phase))
        del masked_image

        plant.display(masked_image_phase,

                      title=title + ' - Masked',
                      dpi=150,
                      fontsize=9,
                      label_y=f'Azimuth Line (x {nlooks_y})',
                      output_file=multilooked_file.replace('.tif', '.png'),
                      label_x=f'Range Bin (x {nlooks_x})',
                      min=-3.14,
                      max=3.14,
                      cmap='jet',
                      no_show=True,

                      force=True)
        del masked_image_phase

        if self.plot_dataset_name:
            title = (f'{self.plot_dataset_name} -'
                     f' RSLC Absolute({pol_1}.{pol_2}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')
        else:
            title = (f'RSLC Absolute({pol_1}.{pol_2}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')

        plant.display(multilooked_file,

                      title=title,
                      dpi=150,
                      fontsize=9,
                      label_y=f'Azimuth Line (x {nlooks_y})',
                      output_file=multilooked_file.replace('.tif', '_mag.png'),
                      label_x=f'Range Bin (x {nlooks_x})',

                      no_show=True,

                      force=True)

        if self.plot_dataset_name:
            title = (f'{self.plot_dataset_name}'
                     f' - RSLC Absolute({pol_1}.{pol_1}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')
        else:
            title = (f'RSLC Absolute({pol_1}.{pol_1}*)'
                     f' Freq. {freq}' + date_str + data_exception_str +
                     ' [rad]')

        plant.filter(pol_1_ref,
                     output_file=beta0_ml_file,
                     square=True,
                     nlooks_x=nlooks_x,
                     nlooks_y=nlooks_y,
                     output_skip_if_existent=self.output_skip_if_existent,
                     force=True)

        plant.display(beta0_ml_file,

                      title=title,
                      dpi=150,
                      fontsize=9,
                      label_y=f'Azimuth Line (x {nlooks_y})',
                      output_file=beta0_ml_file.replace('.tif', '_mag.png'),
                      label_x=f'Range Bin (x {nlooks_x})',

                      no_show=True,

                      force=True)

        plant.save_image(profile_no_offset_phase,
                         output_file=profile_phase_processed_file,
                         metadata=metadata_dict,
                         force=True)

        if self.remove_cross_mul_file and os.path.isfile(cross_mul_file):
            os.remove(cross_mul_file)


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        with PlantIsce3OffDiagAnalysis(parser, argv) as self_obj:
            ret = self_obj.run()
        return ret


def main_cli(*args, **kwargs):
    main(*args, **kwargs)


if __name__ == '__main__':
    main()
