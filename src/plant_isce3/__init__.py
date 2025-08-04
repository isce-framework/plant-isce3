import sys
# import plant

from plant_isce3.plant_isce3_lib import *
import plant_isce3.plant_isce3_lib
# import print_test

__version__ = version = VERSION = '0.0.9'

# import plant_isce3.plant_isce3_info


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class NameWrapper(object):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        try:
            return getattr(self.wrapped, name)
        except AttributeError:
            pass
        # if name == '__version__':
        #     return plant.VERSION
        # if name in alias_dict.keys():
        #     name = alias_dict[name]
        return plant_isce3_lib.ModuleWrapper(name)

    def __dir__(self):
        return plant.__dir__()


sys.modules[__name__] = NameWrapper(sys.modules[__name__])
