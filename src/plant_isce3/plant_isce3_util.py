#!/usr/bin/env python3

import os
import numpy as np
import plant
import plant_isce3
import datetime
import h5py
import isce3
from osgeo import gdal
from nisar.products.readers import open_product


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            cmap=1,
                            default_options=1,
                            output_file=1)

    parser.add_argument('--frequency',
                        dest='frequency',
                        type=str,
                        help='Frequency band, either "A" or "B".')

    parser.add_argument('--data',
                        '--images',
                        dest='data_file',
                        type=str,
                        help=("File where the product's data layers will be"
                              " stored."))

    parser.add_argument('--mask',
                        '--mask-layer',
                        '--save-mask',
                        '--save-mask-layer',
                        dest='mask_file',
                        type=str,
                        help=("File where the product's mask layer will be"
                              " stored."))

    parser.add_argument('--runconfig',
                        '--runconfig-file',
                        dest='runconfig_file',
                        type=str,
                        help=("File where the runconfig used to generate the"
                              " product will be stored."))

    parser.add_argument('--layover-shadow-mask',
                        '--layover-shadow-mask-layer',
                        '--save-layover-shadow-mask',
                        '--save-layover-shadow-mask-layer',
                        dest='layover_shadow_mask_file',
                        type=str,
                        help=("File where the product's mask layer will be"
                              " stored."))

    parser.add_argument('--orbit-kml',
                        '--save-orbit-kml',
                        dest='orbit_kml_file',
                        type=str,
                        help=("KML file where the product's orbit ephemeris"
                              " will be stored."))

    return parser


class PlantIsce3Util(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        nisar_product_obj = open_product(self.input_file)

        if self.frequency is None:
            freq_pol_dict = nisar_product_obj.polarizations
            self.frequency = list(freq_pol_dict.keys())[0]

        if self.orbit_kml_file:
            self.save_orbit_kml(nisar_product_obj)

        if self.mask_file:
            self.save_mask(nisar_product_obj)

        if self.layover_shadow_mask_file:
            self.save_layover_shadow_mask(nisar_product_obj)

        if self.runconfig_file:
            self.save_runconfig_file(nisar_product_obj)

        if self.data_file:
            self.save_data()

    def save_data(self):
        image_ref = f'NISAR:{self.input_file}:{self.frequency}'
        image_obj = self.read_image(image_ref)
        self.save_image(image_obj, output_file=self.data_file)

    def save_mask(self, nisar_product_obj):
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

        mask_ctable = gdal.ColorTable()

        mask_ctable.SetColorEntry(0, (175, 175, 175))

        mask_ctable.SetColorEntry(255, (0, 0, 0))

        if not self.cmap:
            self.cmap = 'viridis'

        mask_array = image_obj.image
        n_subswaths = np.max(mask_array[(mask_array != 255)])
        print('number of subswaths:', n_subswaths)

        for subswath in range(1, n_subswaths + 1):
            color = plant.get_color_display(subswath + 1,
                                            flag_decreasing=True,
                                            n_colors=n_subswaths + 2,
                                            cmap=self.cmap)
            color_rgb = tuple([int(255 * x) for x in color[0:3]])
            mask_ctable.SetColorEntry(subswath, color_rgb)

        self.save_image(image_obj, output_file=self.mask_file,
                        out_null=255, ctable=mask_ctable)

    def save_layover_shadow_mask(self, nisar_product_obj):
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

        layover_shadow_mask_ctable = gdal.ColorTable()

        layover_shadow_mask_ctable.SetColorEntry(0, (175, 175, 175))

        layover_shadow_mask_ctable.SetColorEntry(1, (64, 64, 64))

        layover_shadow_mask_ctable.SetColorEntry(2, (223, 223, 223))

        layover_shadow_mask_ctable.SetColorEntry(3, (0, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(11, (32, 32, 32))

        layover_shadow_mask_ctable.SetColorEntry(13, (0, 128, 128))

        layover_shadow_mask_ctable.SetColorEntry(22, (255, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(23, (128, 255, 255))

        layover_shadow_mask_ctable.SetColorEntry(33, (128, 128, 128))

        layover_shadow_mask_ctable.SetColorEntry(255, (0, 0, 0))

        self.save_image(image_obj, output_file=self.layover_shadow_mask_file,
                        out_null=255, ctable=layover_shadow_mask_ctable)

    def save_runconfig_file(self, nisar_product_obj):

        ret = self.overwrite_file_check(self.runconfig_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        h5_obj = h5py.File(self.input_file, 'r')

        runconfig_path = (f'/science/LSAR/{nisar_product_obj.productType}/'
                          'metadata/processingInformation/'
                          'parameters/runConfigurationContents')

        runconfig_str = str(h5_obj[runconfig_path][()].decode('utf-8'))
        h5_obj.close()
        runconfig_str = runconfig_str.replace("\\n", "\n") + '\n'

        output_dir = os.path.dirname(self.runconfig_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.runconfig_file, "w") as f:

            f.write(runconfig_str)
            f.close()

        print(f'## file saved: {self.runconfig_file} (YAML)')

    def save_orbit_kml(self, nisar_product_obj):

        ret = self.overwrite_file_check(self.orbit_kml_file)
        if not ret:
            self.print('Operation cancelled.', 1)
            return

        orbit = nisar_product_obj.getOrbit()

        h5_obj = h5py.File(self.input_file, 'r')
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
            print('polygon: ', polygon)
            print('reference epoch:', reference_epoch)

        for pos, time in zip(state_vectors_pos, state_vectors_time):
            time_str = str(reference_epoch + isce3.core.TimeDelta(time))
            llh_list.append(ellipsoid.xyz_to_lon_lat(pos))
            time_list.append(time_str)

        output_dir = os.path.dirname(self.orbit_kml_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.orbit_kml_file, 'w') as fp:

            fp.write('<?xml version="1.0" encoding="UTF-8"?> \n')
            fp.write('<kml xmlns="http://www.opengis.net/kml/2.2" ')
            fp.write('xmlns:gx="http://www.google.com/kml/ext/2.2"> \n')
            fp.write('<Document> \n')

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
        if plant.isfile(self.orbit_kml_file) and self.verbose:
            print('## file saved: %s (KML)' % self.orbit_kml_file)

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
