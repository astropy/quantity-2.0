# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class Array API compatibility."""

import astropy.units as u
import numpy as np
import pytest

from quantity import Quantity, api

from .conftest import ARRAY_NAMESPACES


def test_issubclass_api():
    """Test that Quantity is a subclass of api.Quantity and api.QuantityArray."""
    assert issubclass(Quantity, api.Quantity)
    assert issubclass(Quantity, api.QuantityArray)


def test_ndarray():
    """Test that ndarray does not satisfy the Quantity API."""
    assert not issubclass(np.ndarray, api.Quantity)
    assert not isinstance(np.array([1, 2, 3]), api.Quantity)


def test_astropy_quantity():
    """Test that astropy.units.Quantity works with the Quantity API."""
    assert issubclass(u.Quantity, api.Quantity)
    assert isinstance(u.Quantity(np.array([1, 2, 3]), u.m), api.Quantity)


# ------------------------------


@pytest.fixture
def array_and_quantity(request):
    xp = request.param.xp
    value = xp.asarray([1.0, 2.0, 3.0])
    q = Quantity(value, u.m)
    return value, q


@pytest.mark.parametrize("array_and_quantity", ARRAY_NAMESPACES, indirect=True)
class TestIsinstanceAPI:
    """Check Quantities are properly recognized independent of the array type."""

    def test_issubclass_api(self, array_and_quantity):
        v, q = array_and_quantity
        assert not issubclass(type(v), api.Quantity)
        assert not issubclass(type(v), api.QuantityArray)
        assert issubclass(type(q), api.Quantity)
        assert issubclass(type(q), api.QuantityArray)

    def test_isinstance_api(self, array_and_quantity):
        v, q = array_and_quantity
        assert not isinstance(v, api.Quantity)
        assert not isinstance(v, api.QuantityArray)
        assert isinstance(q, api.Quantity)
        assert isinstance(q, api.QuantityArray)
