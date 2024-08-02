import sys
import time

import plant_isce3
import importlib
from collections.abc import Sequence
from nisar.products.readers import open_product
import numpy as np
import isce3

import plant

class PlantIsce3Script(plant.PlantScript):

    def update_geogrid(self, radar_grid, dem_raster, geo=None,
                       slc_obj=None):

        if geo is None:
            geo = isce3.geocode.GeocodeFloat32()
            geo.orbit = slc_obj.getOrbit()
            geo.ellipsoid = isce3.core.Ellipsoid()

        width = self.plant_geogrid_obj.width
        length = self.plant_geogrid_obj.length
        x0_orig = self.plant_geogrid_obj.x0
        y0_orig = self.plant_geogrid_obj.y0
        step_x = self.plant_geogrid_obj.step_x
        step_y = self.plant_geogrid_obj.step_y

        if width is None:
            width = -9999
        if length is None:
            length = -9999

        if x0_orig is None:
            x0_orig = np.nan
        if y0_orig is None:
            y0_orig = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_x):
            step_x = plant.m_to_deg_lon(30.)
        elif step_x is None:
            step_x = np.nan

        if self.epsg == 4326 and not plant.isvalid(step_y):
            step_y = plant.m_to_deg_lat(30.)
        elif step_y is None:
            step_y = np.nan

        geo.geogrid(x0_orig,
                    y0_orig,
                    step_x,
                    step_y,
                    width,
                    length,
                    self.epsg)

        geo.update_geogrid(radar_grid, dem_raster)

        print('*** x0:', self.plant_geogrid_obj.x0)
        print('*** xf:', self.plant_geogrid_obj.xf)
        print('*** y0:', self.plant_geogrid_obj.y0)
        print('*** yf:', self.plant_geogrid_obj.yf)
        print('*** step_x:', self.plant_geogrid_obj.step_x)
        print('*** step_y:', self.plant_geogrid_obj.step_y)
        print('*** length:', self.plant_geogrid_obj.length)
        print('*** width:', self.plant_geogrid_obj.width)
        print('*** self.epsg:', self.epsg)
        projection = plant.epsg_to_wkt(self.epsg)
        print('*** projection:', projection)

        self.plant_geogrid_obj = plant.PlantGeogrid(
            y0=geo.geogrid_start_y,
            length=geo.geogrid_length,
            x0=geo.geogrid_start_x,
            width=geo.geogrid_width,
            step_x=geo.geogrid_spacing_x,
            step_y=-abs(geo.geogrid_spacing_y),
            projection=projection)

        print('*** x0 (updated):', self.plant_geogrid_obj.x0)
        print('*** xf (updated):', self.plant_geogrid_obj.xf)
        print('*** y0 (updated):', self.plant_geogrid_obj.y0)
        print('*** yf (updated):', self.plant_geogrid_obj.yf)
        print('*** step_x (updated):',
              self.plant_geogrid_obj.step_x)
        print('*** step_y (updated):',
              self.plant_geogrid_obj.step_y)

    def _get_input_raster_from_nisar_slc(self, input_raster,
                                         frequency_str=None):

        if frequency_str is None:
            nisar_product_obj = open_product(self.input_file)
            frequency_str = list(nisar_product_obj.polarizations.keys())[0]
            del nisar_product_obj

        if input_raster is not None:
            flag_apply_transformation = \
                self.plant_transform_obj.flag_apply_transformation()
            image_obj = self.read_image(input_raster)

            if flag_apply_transformation:
                temp_file = plant.get_temporary_file(append=True,
                                                     ext='vrt')
                self.print(f'*** creating temporary file: {temp_file}')
                self.save_image(image_obj, temp_file, force=True,
                                output_format='VRT')
                input_raster = temp_file
        else:
            raster_file = f'NISAR:{self.input_file}:{frequency_str}'
            temp_file = plant.get_temporary_file(append=True,
                                                 ext='vrt')
            self.print(f'*** creating temporary file: {temp_file}')
            image_obj = self.read_image(raster_file)
            print('*** image_obj.nbands:', image_obj.nbands)
            self.save_image(image_obj, temp_file, force=True,
                            output_format='VRT')
            input_raster = temp_file
        ret_dict = {}
        ret_dict['input_raster'] = input_raster
        ret_dict['image_obj'] = image_obj
        return ret_dict

    def _get_radar_grid(self, slc_obj, frequency_str):
        radar_grid = slc_obj.getRadarGrid(frequency_str)

        if self.select_row is not None or self.select_col is not None:
            self.plant_transform_obj.update_crop_window(
                length_orig=radar_grid.length,
                width_orig=radar_grid.width)
            y0 = self.plant_transform_obj._offset_y
            if y0 is None:
                y0 = 0
            x0 = self.plant_transform_obj._offset_x
            if x0 is None:
                x0 = 0
            length = self.plant_transform_obj.length
            if length is None:
                length = radar_grid.length
            width = self.plant_transform_obj.width
            if width is None:
                width = radar_grid.width
            radar_grid = radar_grid.offset_and_resize(
                y0, x0, length, width)

        if (self.nlooks_az > 1 or self.nlooks_rg > 1):
            radar_grid_ml = radar_grid.multilook(self.nlooks_az,
                                                 self.nlooks_rg)
        else:
            radar_grid_ml = radar_grid

        return radar_grid_ml

def _get_output_dict_from_parser(parser, args, module_name):
    orig_index = []
    if isinstance(args, dict):
        args_keys = args.keys()
        kwargs = args
    else:
        args_keys = []
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                args_keys.append(arg[2:])
                orig_index.append(i)
            elif arg.startswith('-') and not plant.isnumeric(arg[1:]):
                args_keys.append(arg[1:])
                orig_index.append(i)
        args_keys = [x.replace('-', '_').strip('_')
                     for x in args_keys]
        kwargs = None
    ret = plant.get_args_from_argparser(parser,
                                        store_true_action=False,
                                        store_false_action=False,
                                        store_action=True,
                                        help_action=False,
                                        dest='output_file')
    output_file_keys = [x.replace('-', '_').strip('_')
                        for x in ret]
    output_file_keys.append('output_file')

    ret = plant.get_args_from_argparser(parser,
                                        store_true_action=False,
                                        store_false_action=False,
                                        store_action=True,
                                        help_action=False,
                                        dest='output_dir')
    output_dir_keys = [x.replace('-', '_').strip('_')
                       for x in ret]
    output_dir_keys.append('output_dir')

    ret = plant.get_args_from_argparser(parser,
                                        dest='output_file')
    flag_output = bool(ret)
    output_key = None

    for key in output_file_keys:
        if key not in args_keys:
            continue
        output_key = key
        if isinstance(args, dict):
            break
        output_key_index = orig_index[args_keys.index(output_key)]
        break

    output_str = ''
    output_args = []
    flag_new_mem_output = False

    if flag_output and output_key:
        if kwargs is not None:
            value_str = kwargs[output_key]
        else:
            value_str = args[output_key_index+1]

        output_str = f' {ret[0]} {value_str}'
        output_args.append(ret[0])
        output_args.append(value_str)

    elif (flag_output and
          not any([key in args_keys
                   for key in output_dir_keys]) and
          module_name != 'plant_display'):

        mem_output_str = 'MEM:'+plant.get_temporary_file()
        output_str = f' {ret[0]} {mem_output_str}'
        output_args.append(ret[0])
        output_args.append(mem_output_str)
        flag_new_mem_output = True

    output_dict = {}
    output_dict['output_str'] = output_str
    output_dict['output_args'] = output_args
    output_dict['output_file_keys'] = output_file_keys
    output_dict['output_dir_keys'] = output_dir_keys
    output_dict['output_file_keys'] = output_file_keys
    output_dict['flag_output'] = flag_output
    output_dict['flag_new_mem_output'] = flag_new_mem_output

    return output_dict

def execute(command,
            verbose=True,

            return_time=False,
            ignore_exception=False):

    if not isinstance(command, list):
        command_vector = plant.get_command_vector(command)
    else:
        command_vector = command

    if len(command_vector) == 0 and verbose:
        print('WARNING command not identified: '+command)
        return ['']

    start_time = None
    module_name = command_vector[0]
    argv = command_vector[1:]

    module_name = module_name.replace('.py', '')
    flag_error = False

    module_obj = importlib.import_module('plant_isce3.'+module_name)

    method_to_execute = getattr(module_obj, 'main')

    if plant.plant_config.logger_obj is None:
        sink = plant.PlantLogger()
    else:
        sink = plant.PlantIndent()

    with sink:
        if verbose:
            arguments = plant.get_command_line_from_argv(argv)
            command_line = (f'{module_name}.py {arguments}')
            print(f'PLAnT-ISCE3 {plant_isce3.VERSION} (API) -'
                  f' {command_line}')

        parser_ref = plant.argparse()
        ret = plant.get_args_from_argparser(parser_ref,
                                            store_true_action=True,
                                            store_false_action=True,
                                            store_action=False,
                                            help_action=False,
                                            dest='cli_mode')
        has_bash_flag = any([element in argv for element in ret])
        if not has_bash_flag:
            argv.append('--no-bash')

        argparse_method = getattr(module_obj, 'get_parser')
        parser = argparse_method()

        output_dict = _get_output_dict_from_parser(
            parser, argv, module_name)

        flag_output = output_dict['flag_new_mem_output']

        if output_dict['flag_new_mem_output']:
            argv.extend(output_dict['output_args'])
            argv.extend(['-u', '--ul', '10'])

        original_sys_argv = sys.argv
        sys.argv = [module_name+'.py'] + argv
        flag_error = False
        ret = None
        if return_time:
            start_time = time.time()
        try:
            ret = method_to_execute(argv)
        except SystemExit as e:
            if len(e.args) == 0 or e.args[0] != 0:
                flag_error = True
                error_message = plant.get_error_message()
        finally:
            sys.argv = original_sys_argv

        if (flag_error and not ignore_exception and
                error_message and 'ERROR' in error_message):
            print(error_message)
        elif flag_error and not ignore_exception:
            print('ERROR executing PLAnT module %s: %s.'
                  % (module_name, error_message))
        if return_time:
            ret = time.time() - start_time
        elif flag_output:
            output_ret = plant._get_output_ret_from_plant_config()
            if output_ret is not None:
                ret = output_ret

        if ret is not None:
            ret_str = ('. Returning object class:'
                       f' {ret.__class__.__name__}')
        else:
            ret_str = ''
        if verbose:
            print(f'PLAnT (API-completed) - {module_name}.py {arguments}'
                  f'{ret_str}')

class ModuleWrapper(object):

    def __init__(self, module_name, *args, ref=None, **kwargs):
        self._module_name = module_name
        self._module_obj = None
        self._ref = ref
        self._args = args
        self._kwargs = kwargs
        self._command = None
        self._set_module_obj(self._module_name)

    def _set_module_obj(self, name):
        self._module_name = self._module_name.replace('.py', '')
        if not self._module_name.startswith('plant_isce3'):
            self._module_name = 'plant_isce3_' + self._module_name
        self._module_obj = importlib.import_module(
            f'plant_isce3.{self._module_name}')

    def __call__(self, *args, **kwargs):
        args = list(self._args) + list(args)
        kwargs = dict(self._kwargs, **kwargs)

        self._set_command(*args, **kwargs)
        if self._command is None:
            return

        flag_mute = kwargs.get('flag_mute', None)
        verbose = kwargs.get('verbose', None) and not (flag_mute is True)
        if self._ref is not None:
            ret = self._ref.execute(self._command, verbose=verbose)
        else:
            ret = execute(self._command, verbose=verbose)
        return ret

    def _set_command(self, *args, **kwargs):
        args_str = self._update_args_str(
            args, args_str='')
        args_str = ' -i ' + args_str

        argparse_method = getattr(self._module_obj, 'get_parser')
        parser = argparse_method()
        output_dict = _get_output_dict_from_parser(
            parser, kwargs, self._module_name)
        output_file_keys = output_dict['output_file_keys']
        output_str = output_dict['output_str']
        self._flag_output = output_dict['flag_output']

        kwargs_str = ''
        for key, value in kwargs.items():
            if key in output_file_keys:
                continue
            if isinstance(value, list):
                value_str = ''
                for v in value:
                    if value_str:
                        value_str += ' '
                    if isinstance(v, str) and "'" not in v:
                        value_str += "'"+str(v)+"'"
                    elif isinstance(v, str):
                        value_str += '"'+str(v)+'"'
                    else:
                        value_str += str(v)
            elif (isinstance(value, plant.PlantImage) or
                  isinstance(value, np.ndarray)):
                arg_id = str(id(value))
                plant.plant_config.variables[arg_id] = value
                value_str = f' MEM:{arg_id}'
            elif not isinstance(value, str) or '"' not in value:
                value_str = f'"{value}"'
            else:
                value_str = f"'{value}'"

            kwargs_dest = {}
            if key.startswith('-'):
                kwargs_arg = {'arg': key}
            else:
                key_with_dashes = key.replace('_', '-')
                kwargs_dest['dest'] = key
                if len(key) == 1:
                    kwargs_arg = {'arg': '-'+key_with_dashes}
                else:
                    kwargs_arg = {'arg': '--'+key_with_dashes}
            flag_valid_argument = False
            for kwargs_argparser in [kwargs_dest, kwargs_arg]:
                if flag_valid_argument:
                    continue
                ret = plant.get_args_from_argparser(parser,
                                                    store_true_action=False,
                                                    store_false_action=False,
                                                    store_action=True,
                                                    help_action=False,
                                                    **kwargs_argparser)
                if ret:
                    kwargs_str += f' {ret[0]} {value_str}'
                    flag_valid_argument = True
                    continue

                ret_store_true = plant.get_args_from_argparser(
                    parser,
                    store_true_action=True,
                    store_false_action=False,
                    store_action=False,
                    help_action=False,
                    **kwargs_argparser)
                if ret_store_true and bool(value):
                    kwargs_str += f' {ret_store_true[0]}'
                elif ret_store_true:
                    dest_store_true = plant.get_args_from_argparser(
                        parser,
                        store_true_action=True,
                        store_false_action=False,
                        store_action=False,
                        help_action=False,
                        get_dest=True,
                        **kwargs_argparser)
                    arg_store_false = plant.get_args_from_argparser(
                        parser,
                        store_true_action=False,
                        store_false_action=True,
                        store_action=False,
                        help_action=False,
                        dest=dest_store_true[0])
                    if arg_store_false:
                        kwargs_str += f' {arg_store_false[0]}'

                if ret_store_true:
                    flag_valid_argument = True
                    continue

                ret_store_false = plant.get_args_from_argparser(
                    parser,
                    store_true_action=False,
                    store_false_action=True,
                    store_action=False,
                    help_action=False,
                    **kwargs_argparser)

                if ret_store_false and bool(value):
                    kwargs_str += f' {ret_store_false[0]}'

                elif ret_store_false:
                    dest_store_false = plant.get_args_from_argparser(
                        parser,
                        store_true_action=False,
                        store_false_action=True,
                        store_action=False,
                        help_action=False,
                        get_dest=True,
                        **kwargs_argparser)
                    arg_store_true = plant.get_args_from_argparser(
                        parser,
                        store_true_action=True,
                        store_false_action=False,
                        store_action=False,
                        help_action=False,
                        dest=dest_store_false[0])
                    if arg_store_true:
                        kwargs_str += f' {arg_store_true[0]}'

                if ret_store_false:
                    flag_valid_argument = True

                if flag_valid_argument:
                    continue
            if not flag_valid_argument:
                print(f'ERROR invalid argument: "{key}"')
                return

        self._command = (f'{self._module_name}.py {args_str} {kwargs_str}'
                         f' {output_str}')

    def _update_args_str(self, args, args_str=''):
        for arg in args:
            if isinstance(arg, str):
                args_str += f' {arg}'
                continue
            if (isinstance(arg, Sequence) and
                    all([isinstance(x, str) for x in arg])):
                args_str += self._update_args_str(arg)
                continue

            if not isinstance(arg, plant.PlantImage):
                arg = plant.PlantImage(arg)

            arg_id = str(id(arg))
            plant.plant_config.variables[arg_id] = arg

            args_str += f' MEM:{arg_id}'
        return args_str

