# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class Array API compatibility."""

import astropy.units as u

from quantity import Quantity, api

from .conftest import ARRAY_NAMESPACES


def test_issubclass_api():
    """Test that Quantity is a subclass of api.Quantity and api.QuantityArray."""
    assert issubclass(Quantity, api.Quantity)
    assert issubclass(Quantity, api.QuantityArray)


def test_astropy_quantity():
    assert issubclass(u.Quantity, api.Quantity)
    assert issubclass(u.Quantity, api.QuantityArray)
    aq = u.Quantity(1.0, u.m)
    assert isinstance(aq, api.Quantity)
    assert isinstance(aq, api.QuantityArray)


class IsinstanceAPITests:
    """Check Quantities are properly recognized independent of the array type."""

    # Note: the actual test classes are created at the end

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a = cls.xp.asarray([1.0, 2.0, 3.0])
        cls.q = Quantity(cls.a, u.m)

    def test_issubclass_api(self):
        assert not issubclass(type(self.a), api.Quantity)
        assert not issubclass(type(self.a), api.QuantityArray)
        # The two below Duplicate test_issubclass_api above, but OK to have
        # it more and less explicit.
        assert issubclass(type(self.q), api.Quantity)
        assert issubclass(type(self.q), api.QuantityArray)

    def test_isinstance_api(self):
        assert not isinstance(self.a, api.Quantity)
        assert not isinstance(self.a, api.QuantityArray)
        assert isinstance(self.q, api.Quantity)
        assert isinstance(self.q, api.QuantityArray)


# Create the test classes for the different array types.
for base_setup in ARRAY_NAMESPACES:
    for tests in (IsinstanceAPITests,):
        name = f"Test{tests.__name__}{base_setup.__name__}"
        globals()[name] = type(name, (tests, base_setup), {})
