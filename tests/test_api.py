# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class Array API compatibility."""

import astropy.units as u
import numpy as np

from quantity import Quantity, api


def test_issubclass_api():
    """Test that Quantity is a subclass of api.Quantity and api.QuantityArray."""
    assert issubclass(Quantity, api.Quantity)
    assert issubclass(Quantity, api.QuantityArray)


def test_isintsance_api():
    """Test that Quantity is an instance of api.Quantity and api.QuantityArray."""
    q = Quantity(value=np.array([1, 2, 3]), unit=u.m)
    assert isinstance(q, api.Quantity)
    assert isinstance(q, api.QuantityArray)
