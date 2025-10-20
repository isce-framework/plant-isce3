import sys
from plant_isce3.plant_isce3_lib import *
import plant_isce3.plant_isce3_lib
from . import readers

__version__ = "0.1.8"

version = VERSION = __version__


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

        return plant_isce3_lib.ModuleWrapper(name)

    def __dir__(self):
        return plant.__dir__()


sys.modules[__name__] = NameWrapper(sys.modules[__name__])
