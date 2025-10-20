from .Base import get_hdf5_file_root_path
from .Raw import Raw
from . import (
    GenericProduct,
    get_hdf5_file_product_type,
    GCOV,
    STATIC,
    GSLC,
    RSLC,
)


def open_product(filename: str, root_path: str = None):
    """
    Open NISAR product (HDF5 file), instantianting an object
    of an existing product class (e.g. RSLC, RRSD), if
    defined, or a general product (GeneralProduct) otherwise.

    Parameters
    ----------
    filename : str
        HDF5 filename
    root_path : str (optional)
        Preliminary root path to check before default root
        path list. This option is intended for non-standard products.

    Returns
    -------
    object
        Object derived from the base class
    """

    if root_path is None:
        root_path = get_hdf5_file_root_path(filename, root_path = root_path)

    product_type = get_hdf5_file_product_type(filename, root_path = root_path)

    # set keyword arguments for class constructors
    kwargs = {}
    kwargs['hdf5file'] = filename
    kwargs['_RootPath'] = root_path

    if product_type == 'RSLC':
        return RSLC(**kwargs)
    if product_type == 'GSLC':
        return GSLC(**kwargs)
    if product_type == 'GCOV':
        return GCOV(**kwargs)
    if product_type == 'STATIC':
        return STATIC(**kwargs)
    if product_type == 'RRSD':
        return Raw(**kwargs)
    kwargs['_ProductType'] = product_type
    return GenericProduct(**kwargs)
