# -*- coding: utf-8 -*-
from __future__ import annotations

# import h5py
import journal
import pyre
# import numpy as np

import isce3
# from isce3.core import DateTime, LookSide, speed_of_light
# from isce3.product import GeoGridParameters, RadarGridParameters
from plant_isce3.readers.GenericProduct import (
    GenericProduct,
    get_hdf5_file_product_type,
)
from plant_isce3.readers.Base import open_h5_file


class STATIC(
    GenericProduct,
    family='nisar.productreader.product',
):
    """
    Class for the static layers (STATIC) product
    """

    def __init__(self, **kwds):
        """
        Constructor to initialize product with HDF5 file.
        """

        ###Read base product information like Identification
        super().__init__(**kwds)

        # Set error channel
        self.error_channel = journal.error('GenericSingleSourceL2Product')

        self.identification.productType = \
            get_hdf5_file_product_type(self.filename,
                                       root_path = self.RootPath)

    def getGeoGridProduct(self):
        """
        Returns the GeoGridProduct object for the product.
        """
        return isce3.product.GeoGridProduct(self.filename)

    def getProductLevel(self):
        """
        Returns the product level.
        """
        return "L2"

    @pyre.export
    def getDopplerCentroid(
        self,
        frequency: str | None = None,
    ) -> isce3.core.LUT2d:
        """
        Returns the Doppler centroid for the given frequency.

        Parameters
        ----------
        frequency : "A" or "B" or None, optional
            The frequency letter (either "A" or "B") or None. Returns the LUT of the
            first frequency on the product if None. Defaults to None.

        Returns
        -------
        isce3.core.LUT2d
            The Doppler centroid LUT
        """
        if frequency is None:
            frequency = self._getFirstFrequency()

        doppler_group_path = f'{self.ProductPath}/nativeDoppler'

        doppler_dataset_path = f'{doppler_group_path}/dopplerCentroid'
        zero_doppler_time_dataset_path = (f'{doppler_group_path}/'
                                          'zeroDopplerTime')
        slant_range_dataset_path = f'{doppler_group_path}/slantRange'

        # extract the native Doppler dataset
        with open_h5_file(self.filename, 'r', libver='latest', swmr=True) as fid:

            doppler = fid[doppler_dataset_path][:]
            zeroDopplerTime = fid[zero_doppler_time_dataset_path][:]
            slantRange = fid[slant_range_dataset_path][:]

        dopplerCentroid = isce3.core.LUT2d(xcoord=slantRange,
                                           ycoord=zeroDopplerTime,
                                           data=doppler)

        return dopplerCentroid
