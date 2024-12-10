#!/usr/bin/env python3

import plant
import plant_isce3

import numpy as np
from osgeo import osr

from nisar.products.readers import open_product


def get_parser():

    descr = ('')
    epilog = ''
    parser = plant.argparse(epilog=epilog,
                            description=descr,
                            input_files=1)

    parser.add_argument('--epsg',
                        dest='epsg',
                        type=int,
                        help='EPSG code for output grids.')

    return parser


class PlantIsce3Info(plant_isce3.PlantIsce3Script):

    def __init__(self, parser, argv=None):

        super().__init__(parser, argv)

    def run(self):

        for i, input_file in enumerate(self.input_files):
            print(f'## input {i+1}:', input_file)
            with plant.PlantIndent():
                self._print_nisar_product_info(input_file)

    def _print_nisar_product_info(self, input_file):
        import shapely.wkt
        nisar_product_obj = open_product(input_file)

        print('product type:', nisar_product_obj.productType)
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

            yf = bounds[1]
            y0 = bounds[3]
            x0 = bounds[2]
            xf = bounds[0]
            print('polygon WKT:', polygon)
            print('bounding box:')
            with plant.PlantIndent():
                print('min lat:', yf)
                print('min lon:', x0)
                print('max lat:', y0)
                print('max lon:', xf)
                bbox = plant.get_bbox(x0=x0, xf=xf, y0=y0, yf=yf)
                coord_str = ('PLAnT bbox argument: -b %.16f %.16f %.16f %.16f'
                             % (bbox[0], bbox[1], bbox[2], bbox[3]))
                print(coord_str)

            if self.epsg is None:
                zones_list = []
                for lat in [y0, yf]:
                    for lon in [x0, xf]:
                        zones_list.append(point2epsg(lon, lat))
                vals, counts = np.unique(zones_list, return_counts=True)
                self.epsg = int(vals[np.argmax(counts)])
                print('closest projection EPSG code supported by NISAR:',
                      self.epsg)

            if self.epsg is not None:
                y_min = np.nan
                y_max = np.nan
                x_min = np.nan
                x_max = np.nan
                for lat in [y0, yf]:
                    for lon in [x0, xf]:
                        y, x = lat_lon_to_projected(lat, lon, self.epsg)
                        if plant.isnan(y_min) or y < y_min:
                            y_min = y
                        if plant.isnan(y_max) or y > y_max:
                            y_max = y
                        if plant.isnan(x_min) or x < x_min:
                            x_min = x
                        if plant.isnan(x_max) or x > x_max:
                            x_max = x

                projected_bbox = plant.get_bbox(x0=x_min, xf=x_max, y0=y_max,
                                                yf=y_min)
                coord_str = ('PLAnT bbox argument: -b %.0f %.0f %.0f %.0f'
                             % (projected_bbox[0], projected_bbox[1],
                                projected_bbox[2], projected_bbox[3]))

                print(f'EPSG {self.epsg} coordinates:')
                with plant.PlantIndent():
                    print('min Y:', y_min)
                    print('min X:', x_min)
                    print('max Y:', y_max)
                    print('max X:', x_max)
                    print(coord_str)


def point2epsg(lon, lat):

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
    osr.UseExceptions()

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
