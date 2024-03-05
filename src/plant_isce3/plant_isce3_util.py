#!/usr/bin/env python3

import numpy as np
import plant
import datetime
import h5py
import isce3
from nisar.products.readers import open_product


def get_parser():
    '''
    Command line parser.
    '''
    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1,
                            default_options=1,
                            output_file=1)

    return parser


class PlantIsce3Util(plant.PlantScript):

    def __init__(self, parser, argv=None):
        '''
        class initialization
        '''
        super().__init__(parser, argv)

    def run(self):
        '''
        run main method
        '''

        nisar_product_obj = open_product(self.input_file)
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
        state_vectors_time = orbit.time
        reference_epoch = orbit.reference_epoch

        with plant.PlantIndent():
            print('polygon: ', polygon)
            print('reference epoch:', reference_epoch)

        if not self.output_file:
            print('')
            print('INFO Please provide an output KML file')
            return

        for pos, time in zip(state_vectors_pos, state_vectors_time):
            time_str = str(reference_epoch+isce3.core.TimeDelta(time))
            llh_list.append(ellipsoid.xyz_to_lon_lat(pos))
            time_list.append(time_str)

        kml_file = self.output_file
        with open(kml_file, 'w') as fp:
            # orbit
            fp.write('<?xml version="1.0" encoding="UTF-8"?> \n')
            fp.write('<kml xmlns="http://www.opengis.net/kml/2.2" ')
            fp.write('xmlns:gx="http://www.google.com/kml/ext/2.2"> \n')
            fp.write('<Document> \n')
            # self.add_polygon(fp, polygon)
            for i in range(len(llh_list)-1):
                self.add_line(fp, orbit, time_list[i:i+2],
                              llh_list[i:i+2],
                              flag_altitude=False, color='#ff000000')
                self.add_line(fp, orbit, time_list[i:i+2],
                              llh_list[i:i+2],
                              flag_altitude=True, color='#ff00ffff')
            fp.write('</Document> \n')
            fp.write('</kml>\n')
        if plant.isfile(kml_file) and self.verbose:
            print('## file saved: %s (KML)' % kml_file)

    def add_polygon(self, fp, polygon):
        fp.write('<Placemark>\n')
        fp.write('    <name>Swath Polygon</name>\n')
        fp.write('    <Polygon>\n')
        fp.write('      <extrude>1</extrude>\n')
        fp.write('      <altitudeMode>relativeToGround</altitudeMode>\n')
        fp.write('      <outerBoundaryIs>\n')
        fp.write('        <LinearRing>\n')
        fp.write('          <coordinates>\n')
        # height = 0
        for vertex in polygon:
            lon, lat, height = vertex
            fp.write(f'            {lon},{lat},{height} \n')
        fp.write('          </coordinates>\n')
        fp.write('        </LinearRing>\n')
        fp.write('      </outerBoundaryIs>\n')
        fp.write('    </Polygon>\n')
        fp.write('  </Placemark>\n')

    def add_line(self, fp, orbit, time_list,
                 llh_list, flag_altitude=True, color=None):
        fp.write('<Folder> \n')
        fp.write('<Placemark> \n')
        fp.write('<gx:Track> \n')

        # time
        for time_str in time_list:
            fp.write(f'  <when>{time_str}</when>\n')

        # coordinates
        for llh in llh_list:
            lon = np.degrees(llh[0])
            lat = np.degrees(llh[1])
            h = llh[2] if flag_altitude else 0
            fp.write(f'   <gx:coord> {lon},{lat},{h}</gx:coord> \n')
        if flag_altitude:
            fp.write('   <altitudeMode>absolute</altitudeMode>\n')

        # annotations
        fp.write('<ExtendedData>\n')
        fp.write('<SchemaData schemaUrl="#schema">\n')
        fp.write('<gx:SimpleArrayData name="UTC time" kml:name="string">\n')
        for time_str in time_list:
            fp.write(f'<gx:value>{time_str}</gx:value>\n')
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
