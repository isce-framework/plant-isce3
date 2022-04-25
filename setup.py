import os
from setuptools import setup

__version__ = version = VERSION = '0.1'

long_description = (
    'PLAnT-ISCE3 is a general-purpose toolbox that uses the open-source Polarimetric'
    ' Interferometric Lab and Analysis Tool (PLAnT) framework to provide an “easy-to-use”'
    ' command-line interface (CLI) for the open-source InSAR Scientific Computing Environment 3'
    ' (ISCE3) framework and leverage ISCE3 capabilities. PLAnT-ISCE3 delivers an interface to'
    ' ISCE3 modules/functionalities focusing on the end-user. Additionally, since most ISCE3'
    ' modules can only be accessed externally via ISCE3 C++ or Python application programming'
    ' interfaces (APIs), i.e., not through ISCE3 command-line interfaces (CLI), PLAnT-ISCE3'
    ' provides unique access to many ISCE3 functionalities that are not directly exposed to'
    ' the end-user.')

package_data_dict = {}

setup(
    name='PLAnT-ISCE3',
    version=version,
    description='PLAnT-ISCE3: Polarimetric Interferometric Lab and Analysis Tool (PLAnT)'
                ' scripts for the InSAR Scientific Computing Environment 3 (ISCE3)',
    package_dir={},
    packages=[],
    package_data=package_data_dict,
    classifiers=['Programming Language :: Python',],
    scripts=['bin/plant_isce3_geocode.py',
    	     'bin/plant_isce3_info.py',
             'bin/plant_isce3_polygon.py',
             'bin/plant_isce3_topo.py'],
    install_requires=['numpy', 'osgeo'],
    url='https://github.com/isce-framework/PLAnT-ISCE3',
    author='Gustavo H. X. Shiroma, Marco Lavalle',
    author_email=('gustavo.h.shiroma@jpl.nasa.gov'),
    license='Copyright by the California Institute of Technology.'
    ' ALL RIGHTS RESERVED.',
    long_description=long_description,
)
