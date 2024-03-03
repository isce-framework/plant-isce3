#!/usr/bin/env python3

import os
import plant
# from osgeo import gdal
import numpy as np
from osgeo import osr
# import isce3
from nisar.products.readers import open_product


def get_parser():
    '''
    Command line parser.
    '''
    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_file=1)

    parser.add_argument('--epsg',
                        dest='epsg',
                        type=int,
                        help='EPSG code for output grids.')

    return parser


class PlantIsce3Info(plant.PlantScript):

    def __init__(self, parser, argv=None):
        '''
        class initialization
        '''
        super().__init__(parser, argv)

    def run(self):
        '''
        run main method
        '''
        self._get_coordinates_from_h5_file(self.input_file)

    def _get_coordinates_from_h5_file(self, input_file):
        import shapely.wkt
        nisar_product_obj = open_product(input_file)
        # print(nisar_product_obj.__dir__())
        print('## product type:', nisar_product_obj.productType)
        print('SAR band:', nisar_product_obj.sarBand)
        print('level:', nisar_product_obj.getProductLevel())
        print('frequencies/polarizations:')
        freq_pol_dict = nisar_product_obj.polarizations
        with plant.PlantIndent():
            for freq, pol_list in freq_pol_dict.items():
                print(f'{freq}: {pol_list}')
        polygon = nisar_product_obj.identification.boundingPolygon
        print('bounding polygon:')
        with plant.PlantIndent():
            bounds = shapely.wkt.loads(polygon).bounds
            lat_arr = [bounds[1], bounds[3]]
            lon_arr = [bounds[2], bounds[0]]
            print('polygon WKT:', polygon)
            print('bounding box:')
            with plant.PlantIndent():
                print('min lat:', lat_arr[0])
                print('min lon:', lon_arr[0])
                print('max lat:', lat_arr[1])
                print('max lon:', lon_arr[1])
                bbox = plant.get_bbox(lat_arr, lon_arr)
                coord_str = ('PLAnT bbox parameter: -b %.16f %.16f %.16f %.16f'
                             % (bbox[0], bbox[1], bbox[2], bbox[3]))
                print(coord_str)

            if self.epsg is None:
                zones_list = []
                for i in range(2):
                    for j in range(2):
                        zones_list.append(point2epsg(lon_arr[i], lat_arr[j]))
                vals, counts = np.unique(zones_list, return_counts=True)
                self.epsg = int(vals[np.argmax(counts)])
                print('closest projection EPSG code supported by NISAR:',
                      self.epsg)

            if self.epsg is not None:
                y_min = np.nan
                y_max = np.nan
                x_min = np.nan
                x_max = np.nan
                for i in range(2):
                    for j in range(2):
                        y, x = lat_lon_to_projected(lat_arr[i], lon_arr[j],
                                                    self.epsg)
                        if plant.isnan(y_min) or y < y_min:
                            y_min = y
                        if plant.isnan(y_max) or y > y_max:
                            y_max = y
                        if plant.isnan(x_min) or x < x_min:
                            x_min = x
                        if plant.isnan(x_max) or x > x_max:
                            x_max = x

                projected_lat_arr = [y_min, y_max]
                projected_lon_arr = [x_min, x_max]
                projected_bbox = plant.get_bbox(projected_lat_arr,
                                                projected_lon_arr)
                coord_str = ('bbox parameter: -b %.0f %.0f %.0f %.0f'
                             % (projected_bbox[0], projected_bbox[1],
                                projected_bbox[2], projected_bbox[3]))

                print(f'EPSG {self.epsg} coordinates:')
                with plant.PlantIndent():
                    print('min Y:', projected_lat_arr[0])
                    print('min X:', projected_lon_arr[0])
                    print('max Y:', projected_lat_arr[1])
                    print('max X:', projected_lon_arr[1])
                    print(coord_str)


def point2epsg(lon, lat):
    """
     Return EPSG code base on a point
     latitude/longitude coordinates
    """
    if lon >= 180.0:
        lon = lon - 360.0
    if lat >= 60.0:
        return 3413
    elif lat <= -60.0:
        return 3031
    elif lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(
        'Could not determine projection for {0},{1}'.format(lat, lon))


def lat_lon_to_projected(north, east, epsg):
    wgs84_coordinate_system = osr.SpatialReference()
    wgs84_coordinate_system.SetWellKnownGeogCS("WGS84")
    try:
        wgs84_coordinate_system.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass

    projected_coordinate_system = osr.SpatialReference()
    projected_coordinate_system.ImportFromEPSG(epsg) 
    try:
        projected_coordinate_system.SetAxisMappingStrategy(
            osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass

    transformation = osr.CoordinateTransformation(wgs84_coordinate_system,
                                                  projected_coordinate_system)
    x, y, _ = transformation.TransformPoint(float(east), float(north), 0)
    return (y, x)


def main(argv=None):
    with plant.PlantLogger():
        parser = get_parser()
        self_obj = PlantIsce3Info(parser, argv)
        ret = self_obj.run()
        return ret


if __name__ == '__main__':
    main()
