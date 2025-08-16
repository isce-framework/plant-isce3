#!/usr/bin/env python3

import os
import shutil

import numpy as np
import plant
import plant_isce3
import datetime
import h5py
import isce3

from nisar.products.readers import open_product

POL_LIST = ['HH', 'HV', 'VH', 'VV', 'RH', 'RV']


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            dem_file=1,
                            cmap=1,
                            default_options=1,
                            output_file=1)

    plant_isce3.add_arguments(parser,
                              burst_ids=1,
                              orbit_files=1,
                              frequency=1)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--cp-pol',
                       '--copy-pol',
                       dest='copy_pol',
                       nargs=2,
                       type=str,
                       help=('Copy polarization. Provide source and'
                             ' destination pols.'))

    group.add_argument('--mv-pol',
                       '--move-pol',
                       '--rename-pol',
                       dest='move_pol',
                       nargs=2,
                       type=str,
                       help=('Rename polarization. Provide source and'
                             ' destination pols.'))

    group.add_argument('--rm-pol',
                       '--remove-pol',
                       dest='remove_pol',
                       type=str,
                       help=('Remove polarization.'))

    group.add_argument('--data',
                       '--images',
                       dest='data_file',
                       action='store_true',

                       help=("Extract product's imagery"))

    group.add_argument('--mask',
                       '--mask-layer',

                       dest='mask_file',
                       action='store_true',

                       help=("Extract mask layer."))

    group.add_argument('--runconfig',
                       '--runconfig-file',
                       dest='runconfig_file',
                       action='store_true',

                       help=("Extract the runconfig used to generate the"
                             " product from its metadata."))

    group.add_argument('--layover-shadow-mask',
                       '--layover-shadow-mask-layer',

                       dest='layover_shadow_mask_file',
                       action='store_true',

                       help=("Extract the layover/shadow mask from the"
                             " product"))

    group.add_argument('--orbit-kml',

                       dest='orbit_kml_file',
                       action='store_true',

                       help=("Save a KML file containing the product's orbit"
                             " ephemeris"))

    group.add_argument('--slant-range',
                       '--slant-range-file',
                       dest='slant_range_file',
                       action='store_true',
                       help=("Save file containing slant-range indexes."
                             " Only available for slant-range products"
                             " (level 1)"))

    group.add_argument('--azimuth-time',
                       '--azimuth-time-file',
                       dest='azimuth_time_file',
                       action='store_true',
                       help=("Save file containing azimuth times."
                             " Only available for slant-range products"
                             " (level 1)"))

    parser.add_argument('--no-bistatic-delay-correction',
                        dest='apply_bistatic_delay_correction',
                        default=True,
                        action='store_false',
                        help=("Prevent the bistatic delay to be applied"))

    parser.add_argument('--no-tropospheric-delay-correction',
                        dest='apply_static_tropospheric_delay_correction',
                        default=True,
                        action='store_false',
                        help=(""))

    parser.add_argument('--beta0',
                        dest='flag_output_complex',
                        default=True,
                        action='store_false',
                        help=("Prevent the static tropospheric delay to be"
                              " applied"))

    parser.add_argument(
        '--no-thermal-correction',
        dest='flag_thermal_correction',
        default=True,
        action='store_false',
        help=("Prevent thermal noise correction to be applied"))

    parser.add_argument('--no-abs-rad-correction',
                        dest='flag_apply_abs_rad_correction',
                        default=True,
                        action='store_false',
                        help=(""))

    return parser


def overwrite_dataset_check(element_name, force=None, element_str='file'):

    if plant.plant_config.flag_all or force:
        return True
    if plant.plant_config.flag_never:
        return False
    while 1:
        res = plant.get_keys(f'The {element_str} {element_name} already'
                             ' exists. Would you like to overwrite'
                             ' it? ([y]es/[n]o)/[A]ll/[N]one ')
        if res == 'n':
            return False
        elif res == 'N':
            plant.plant_config.flag_never = True
            return False
        elif res == 'y':
            return True
        elif res == 'A':
            plant.plant_config.flag_all = True
            return True


class PlantIsce3Util(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        ret = self.overwrite_file_check(self.output_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        if (not self.input_file.endswith('.h5') and
                not self.input_file.endswith('.nc') and
                not self.input_file.endswith('.SAFE') and
                not self.input_file.endswith('.zip')):

            self.run_raster_as_input()
            plant.append_output_file(self.output_file)
            return self.output_file

        plant_product_obj = self.load_product()
        if self.orbit_kml_file:
            self.save_orbit_kml(plant_product_obj)

        elif self.slant_range_file:
            self.save_slant_range_file(plant_product_obj)

        elif self.azimuth_time_file:
            self.save_azimuth_time_file(plant_product_obj)

        elif (plant_product_obj.sensor_name == 'Sentinel-1'):
            self.run_sentinel_1_as_input(plant_product_obj)

        else:
            self.run_nisar_as_input()

        plant.append_output_file(self.output_file)
        return self.output_file

    def run_sentinel_1_as_input(self, plant_product_obj):

        flag_output_complex = self.flag_output_complex
        flag_thermal_correction = self.flag_thermal_correction
        flag_apply_abs_rad_correction = self.flag_apply_abs_rad_correction

        input_raster = plant_product_obj.get_sentinel_1_input_raster(
            flag_output_complex=flag_output_complex,
            flag_thermal_correction=flag_thermal_correction,
            flag_apply_abs_rad_correction=flag_apply_abs_rad_correction)

        image_obj = plant.read_image(input_raster)

        if self.mask_file:
            raise NotImplementedError
        elif self.layover_shadow_mask_file:
            raise NotImplementedError
        elif self.runconfig_file:
            raise NotImplementedError

        self.save_data(image_obj=image_obj)

    def run_nisar_as_input(self):
        nisar_product_obj = open_product(self.input_file)

        if self.frequency is None:
            freq_pol_dict = nisar_product_obj.polarizations
            self.frequency = list(freq_pol_dict.keys())[0]

        if self.mask_file:
            self.save_mask(nisar_product_obj)

        elif self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(nisar_product_obj)

        elif self.data_file:
            self.save_data()

        elif self.runconfig_file:
            self.save_runconfig_file(nisar_product_obj)

        else:
            if self.input_file != self.output_file:

                input_file_obj = h5py.File(self.input_file, 'r')
                input_file_obj.close()
                shutil.copyfile(self.input_file, self.output_file)

            if self.copy_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.copy_pol[0],
                                            output_pol=self.copy_pol[1])
                    print('done')

            if self.remove_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.remove_pol)
                    print('done')

            if self.move_pol:
                with h5py.File(self.output_file, 'a') as root_ds:
                    self.copy_pol_recursive(root_ds, key='/',
                                            input_pol=self.move_pol[0],
                                            output_pol=self.move_pol[1])
                    self.remove_pol_recursive(root_ds, key='/',
                                              pol=self.move_pol[0])
                    print('done')

            print(f'# file saved: {self.output_file}')

    def run_raster_as_input(self):
        image_obj = plant.read_image(self.input_file)

        if self.mask_file:
            self.save_mask(image_obj=image_obj)

        elif self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(image_obj=image_obj)

        elif self.data_file:
            self.save_data(image_obj=image_obj)

        else:
            self.save_image(image_obj, self.output_file)

    def get_az_rg_timing_correction_luts(self, plant_product_obj, radar_grid):
        rg_step_meters = 10 * radar_grid.range_pixel_spacing
        az_step_meters = rg_step_meters
        apply_bistatic_delay_correction = \
            self.apply_bistatic_delay_correction
        apply_static_tropospheric_delay_correction = \
            self.apply_static_tropospheric_delay_correction

        dem_raster = plant_isce3.get_isce3_raster(self.dem_file)

        rg_lut, az_lut = plant_isce3.compute_correction_lut(
            plant_product_obj.burst,
            dem_raster,

            rg_step_meters,
            az_step_meters,
            apply_bistatic_delay_correction,
            apply_static_tropospheric_delay_correction)

        return rg_lut, az_lut

    def save_slant_range_file(self, plant_product_obj):

        radar_grid = plant_product_obj.get_radar_grid()
        new_var_array = np.repeat(
            [radar_grid.slant_ranges], radar_grid.length, axis=0)

        if plant_product_obj.sensor_name == 'Sentinel-1':
            rg_lut, _ = self.get_az_rg_timing_correction_luts(
                plant_product_obj, radar_grid)
        else:
            rg_lut = None

        if rg_lut is not None:
            rg_lut.bounds_error = False

            for i in range(radar_grid.length):
                range_shift = \
                    rg_lut.eval(radar_grid.sensing_times[i],
                                radar_grid.slant_ranges)

                new_var_array[i, :] = new_var_array[i, :] + range_shift

        plant_image_obj = plant.PlantImage(new_var_array)
        plant_image_obj.set_name('Slant-range Distance in Meters')
        self.save_image(plant_image_obj, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_azimuth_time_file(self, plant_product_obj):
        radar_grid = plant_product_obj.get_radar_grid()
        new_var_array = np.repeat(np.transpose(
            [radar_grid.sensing_times]),
            radar_grid.width, axis=1)

        if plant_product_obj.sensor_name == 'Sentinel-1':
            _, az_lut = self.get_az_rg_timing_correction_luts(
                plant_product_obj, radar_grid)
        else:
            az_lut = None

        if az_lut is not None:
            az_lut.bounds_error = False

            for i in range(radar_grid.length):
                for j in range(radar_grid.width):
                    azimuth_delay = \
                        az_lut.eval(radar_grid.sensing_times[i],
                                    radar_grid.slant_ranges[j])

                    new_var_array[i, j] = new_var_array[i, j] + azimuth_delay

        plant_image_obj = plant.PlantImage(new_var_array)

        ref_epoch = str(radar_grid.ref_epoch)
        plant_image_obj.set_name(f'Azimuth Time in Seconds Since {ref_epoch}')
        plant_image_obj.set_metadata(f'REF_EPOCH: {ref_epoch}[s]')

        self.save_image(plant_image_obj, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_data(self, image_obj=None):
        if image_obj is None:
            image_ref = f'NISAR:{self.input_file}:{self.frequency}'
            image_obj = self.read_image(image_ref)

        if ('complex' not in plant.get_dtype_name(image_obj.dtype).lower() or
                self.flag_output_complex is not False):
            self.save_image(image_obj, output_file=self.output_file)
            plant.append_output_file(self.output_file)
            return

        image_list = []
        for b in range(image_obj.nbands):
            image_list.append(
                np.absolute(image_obj.get_image(band=b)) ** 2)

        self.save_image(image_list, output_file=self.output_file)
        plant.append_output_file(self.output_file)

    def save_mask(self, nisar_product_obj=None,
                  image_obj=None):

        if image_obj is None:
            if nisar_product_obj.productType not in ['GSLC', 'GCOV']:
                error_msg = (f'ERROR cannot save mask for product type'
                             f' "{nisar_product_obj.productType}".'
                             ' Not implemented.')
                print(error_msg)
                raise NotImplementedError(error_msg)

            grid_path = (f'{nisar_product_obj.GridPath}'
                         f'/frequency{self.frequency}/mask')
            image_ref = f'NETCDF:{self.input_file}:{grid_path}'

            image_obj = self.read_image(image_ref)

        mask_array = image_obj.image

        mask_ctable = self.get_mask_ctable(mask_array)

        self.save_image(image_obj, output_file=self.output_file,
                        out_null=255, ctable=mask_ctable)
        plant.append_output_file(self.output_file)

    def save_layover_shadow_mask(self, nisar_product_obj=None,
                                 image_obj=None):

        if image_obj is None:
            if nisar_product_obj.productType not in ['GCOV']:
                error_msg = (f'ERROR cannot save mask for product type'
                             f' "{nisar_product_obj.productType}".'
                             ' Not implemented.')
                print(error_msg)
                raise NotImplementedError(error_msg)

            grid_path = (f'{nisar_product_obj.GridPath}'
                         f'/frequency{self.frequency}/layoverShadowMask')
            image_ref = f'NETCDF:{self.input_file}:{grid_path}'
            image_obj = self.read_image(image_ref)

        layover_shadow_mask_ctable = self.get_layover_shadow_mask_ctable()

        self.save_image(image_obj, output_file=self.output_file,
                        out_null=255, ctable=layover_shadow_mask_ctable)
        plant.append_output_file(self.output_file)

    def save_runconfig_file(self, nisar_product_obj):

        h5_obj = h5py.File(self.input_file, 'r')

        runconfig_path = (f'/science/LSAR/{nisar_product_obj.productType}/'
                          'metadata/processingInformation/'
                          'parameters/runConfigurationContents')

        runconfig_str = str(h5_obj[runconfig_path][()].decode('utf-8'))
        h5_obj.close()
        runconfig_str = runconfig_str.replace("\\n", "\n") + '\n'

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, "w") as f:

            f.write(runconfig_str)
            f.close()

        print(f'## file saved: {self.output_file} (YAML)')
        plant.append_output_file(self.output_file)

    def save_orbit_kml(self, plant_product_obj):

        orbit = plant_product_obj.get_orbit()
        flag_has_polygon = False

        if plant_product_obj.sensor_name == 'NISAR':

            h5_obj = h5py.File(self.input_file, 'r')

            flag_has_polygon = True
            polygon_dataset = '//science/LSAR/identification/boundingPolygon'
            polygon_str = str(h5_obj[polygon_dataset][()].decode('utf-8'))
            h5_obj.close()
            polygon_str = polygon_str.replace('POLYGON', '')
            polygon_str_ref = ''
            while polygon_str_ref != polygon_str:
                polygon_str_ref = polygon_str
                polygon_str = polygon_str.replace('(', '')
            polygon_str_ref = ''
            while polygon_str_ref != polygon_str:
                polygon_str_ref = polygon_str
                polygon_str = polygon_str.replace(')', '')
            polygon = polygon_str.split(',')
            polygon = [p.strip().split(' ') for p in polygon]

        ellipsoid = isce3.core.Ellipsoid()
        time_list = []
        llh_list = []
        state_vectors_pos = orbit.position
        state_vectors_vel = orbit.velocity
        state_vectors_time = orbit.time
        reference_epoch = orbit.reference_epoch

        with plant.PlantIndent():
            if flag_has_polygon:
                print('polygon: ', polygon)
            print('reference epoch:', reference_epoch)

        for pos, time in zip(state_vectors_pos, state_vectors_time):
            time_str = str(reference_epoch + isce3.core.TimeDelta(time))
            llh_list.append(ellipsoid.xyz_to_lon_lat(pos))
            time_list.append(time_str)

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, 'w') as fp:

            fp.write('<?xml version="1.0" encoding="UTF-8"?> \n')
            fp.write('<kml xmlns="http://www.opengis.net/kml/2.2" ')
            fp.write('xmlns:gx="http://www.google.com/kml/ext/2.2"> \n')
            fp.write('<Document> \n')

            if flag_has_polygon:
                self.add_polygon(fp, polygon)

            self.add_line(fp, state_vectors_pos, state_vectors_vel, time_list,
                          llh_list,
                          flag_altitude=False, color='#ff000000')
            self.add_line(fp, state_vectors_pos, state_vectors_vel, time_list,
                          llh_list,
                          flag_altitude=True, color='#ff00ffff')
            fp.write('<PolyStyle>')
            fp.write('  <color>7f0000ff</color>')
            fp.write('  <colorMode>normal</colorMode>')
            fp.write('  <fill>1</fill>')
            fp.write('  <outline>1</outline>')
            fp.write('</PolyStyle>')
            fp.write('<Schema id="schema">\n')
            fp.write('  <gx:SimpleArrayField name="UTC time" type="string">\n')
            fp.write('    <displayName>X</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="X" type="float">\n')
            fp.write('    <displayName>X</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="Y" type="float">\n')
            fp.write('    <displayName>Y</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('  <gx:SimpleArrayField name="Z" type="float">\n')
            fp.write('    <displayName>Z</displayName>\n')
            fp.write('  </gx:SimpleArrayField>\n')
            fp.write('</Schema>\n')
            fp.write('</Document> \n')
            fp.write('</kml>\n')
        if plant.isfile(self.output_file) and self.verbose:
            print('## file saved: %s (KML)' % self.output_file)
        plant.append_output_file(self.output_file)

    def add_polygon(self, fp, polygon):
        fp.write('<Placemark>\n')
        fp.write('    <name>Swath Polygon</name>\n')
        fp.write('    <Polygon>\n')
        fp.write('      <extrude>1</extrude>\n')
        fp.write('      <altitudeMode>relativeToGround</altitudeMode>\n')
        fp.write('      <outerBoundaryIs>\n')
        fp.write('        <LinearRing>\n')
        fp.write('          <coordinates>\n')

        for vertex in polygon:
            if len(vertex) == 3:
                lon, lat, height = vertex
            else:
                lon, lat = vertex
                height = 0
            fp.write(f'            {lon},{lat},{height} \n')
        fp.write('          </coordinates>\n')
        fp.write('        </LinearRing>\n')
        fp.write('      </outerBoundaryIs>\n')
        fp.write('    </Polygon>\n')
        fp.write('  </Placemark>\n')

    def add_line(self, fp, state_vectors_pos, state_vectors_vel,
                 time_list,
                 llh_list, flag_altitude=True, color=None):
        fp.write('<Folder> \n')

        if flag_altitude:
            fp.write('<name>Platform Track</name>\n')
        else:
            fp.write('<name>Platform Ground Track</name>\n')

        fp.write('<kml:visibility>0</kml:visibility>')
        fp.write('<visibility>0</visibility>')

        for i, (llh, time_str) in enumerate(zip(llh_list, time_list)):

            fp.write('<Placemark>\n')
            if flag_altitude:
                fp.write(f'  <name>t{i}</name>\n')
            else:
                fp.write(f'  <name>g{i}</name>\n')

            description = f'UTC time: {time_str} \n'
            for axis, coord_str in enumerate(['X', 'Y', 'Z']):
                description += (f'    pos. {coord_str}:'
                                f' {state_vectors_pos[i][axis]}\n')
            for axis, coord_str in enumerate(['X', 'Y', 'Z']):
                description += (f'    vel. {coord_str}:'
                                f' {state_vectors_vel[i][axis]}\n')
            fp.write(f'  <description>{description}</description>\n')
            fp.write('  <Point>\n')

            lon = np.degrees(llh[0])
            lat = np.degrees(llh[1])
            h = llh[2] if flag_altitude else 0
            fp.write(f'   <coordinates> {lon}, {lat}, {h}</coordinates> \n')
            if flag_altitude:
                fp.write('   <altitudeMode>absolute</altitudeMode>\n')
            fp.write('  </Point>\n')
            fp.write('</Placemark>\n')

        fp.write('</Folder><Folder> \n')
        if flag_altitude:
            fp.write('<name>Platform Track</name>\n')
        else:
            fp.write('<name>Platform Ground Track</name>\n')

        fp.write('<Placemark> \n')
        if flag_altitude:
            fp.write('<name>Platform Track</name>')
        else:
            fp.write('<name>Platform Ground Track</name>')

        fp.write('<gx:Track> \n')

        for time_str in time_list:
            fp.write(f'  <when>{time_str}</when>\n')

        for llh in llh_list:
            lon = np.degrees(llh[0])
            lat = np.degrees(llh[1])
            h = llh[2] if flag_altitude else 0
            fp.write(f'   <gx:coord> {lon},{lat},{h}</gx:coord> \n')

        if flag_altitude:
            fp.write('   <altitudeMode>absolute</altitudeMode>\n')

        fp.write('<ExtendedData>\n')
        fp.write('<SchemaData schemaUrl="#schema">\n')

        fp.write('<gx:SimpleArrayData name="UTC time" kml:name="string">\n')
        for time_str in time_list:
            fp.write(f'<gx:value>{time_str}</gx:value>\n')
        fp.write('</gx:SimpleArrayData>\n')

        for e, coord_str in enumerate(['X', 'Y', 'Z']):
            fp.write(
                f'<gx:SimpleArrayData name="{coord_str}" kml:name="float">\n')
            for i, time_str in enumerate(time_list):
                fp.write(f'<gx:value>{state_vectors_pos[i][e]}</gx:value>\n')
            fp.write('</gx:SimpleArrayData>\n')

        fp.write('</SchemaData>\n')
        fp.write('</ExtendedData>\n')
        fp.write('</gx:Track>\n')
        fp.write('</Placemark> \n')
        fp.write('</Folder> \n')

    def copy_pol_recursive(self, hdf_obj: h5py.Group, key: str, input_pol: str,
                           output_pol: str):

        h5_element = hdf_obj[key]

        if key == input_pol:
            input_path = f'{hdf_obj.name}/{input_pol}'
            output_path = f'{hdf_obj.name}/{output_pol}'
            if output_path in hdf_obj:

                if isinstance(h5_element, h5py.Group):
                    element_str = 'H5 group'
                else:
                    element_str = 'H5 dataset'

                ret = overwrite_dataset_check(output_path, force=self.force,
                                              element_str=element_str)

                if ret is False:
                    return

                del hdf_obj[output_pol]

            print(f'copying" "{input_path}"')
            print(f'     to" "{output_path}"')
            hdf_obj.copy(input_path, hdf_obj, output_path)
            return

        if isinstance(h5_element, h5py.Group):

            for sub_key in h5_element.keys():
                self.copy_pol_recursive(h5_element, sub_key, input_pol,
                                        output_pol)

        elif input_pol in POL_LIST and key == 'listOfPolarizations':
            list_of_polarizations = h5_element[()]
            print('input list_of_polarizations:', list_of_polarizations)
            flag_input_pol_found = any([pol.decode() == input_pol
                                        for pol in list_of_polarizations])

            if not flag_input_pol_found:
                print(f'WARNING input pol {input_pol} not found in the list'
                      f'of polarization at: "{h5_element.name}"')
                return
            flag_output_pol_found = any([pol.decode() == output_pol
                                        for pol in list_of_polarizations])

            if flag_output_pol_found:
                print(f'WARNING output pol {output_pol} already exists in list'
                      f'of polarization at: "{h5_element.name}"')
                return
            list_of_polarizations = np.sort(np.append(list_of_polarizations,
                                            np.bytes_(output_pol)))
            print('output list_of_polarizations:', list_of_polarizations)
            del hdf_obj[key]
            hdf_obj.create_dataset(key, data=list_of_polarizations)

    def remove_pol_recursive(self, hdf_obj: h5py.Group, key: str, pol: str):

        h5_element = hdf_obj[key]

        if key == pol:
            path = f'{hdf_obj.name}/{pol}'
            del hdf_obj[path]
            return

        if isinstance(h5_element, h5py.Group):

            for sub_key in h5_element.keys():
                self.remove_pol_recursive(h5_element, sub_key, pol)

        elif pol in POL_LIST and key == 'listOfPolarizations':
            list_of_polarizations = h5_element[()]
            list_of_polarizations_decoded = \
                [p.decode() for p in list_of_polarizations]

            print('input list_of_polarizations:',
                  list_of_polarizations_decoded)

            if pol not in list_of_polarizations_decoded:
                print('*** flag_pol_found. Skipping.')
                return

            list_of_polarizations_decoded.remove(pol)
            print('output list_of_polarizations:',
                  list_of_polarizations_decoded)
            del hdf_obj[key]
            hdf_obj.create_dataset(
                key, data=np.bytes_(list_of_polarizations_decoded))


def get_datetime_from_isoformat(ref_epoch):
    ref_epoch = datetime.datetime.strptime(
        ref_epoch.isoformat().split('.')[0],
        "%Y-%m-%dT%H:%M:%S")
    return ref_epoch


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Util(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
