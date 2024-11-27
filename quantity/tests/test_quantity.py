# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class, creation and basic methods.

Note: tests classes are combined with setups for different array types
at the very end.  Hence, they do not have the usual Test prefix.
"""

from __future__ import annotations

import copy

import array_api_compat
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quantity import Quantity

from .conftest import ARRAY_NAMESPACES


class QuantityCreationTests:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.value = cls.xp.arange(10.0)
        cls.unit = u.m
        cls.ARRAY_CLASS = type(cls.value)

    def test_setup(self):
        try:
            xp = self.value.__array_namespace__()
        except AttributeError:
            xp = array_api_compat.array_namespace(self.value)

        assert self.xp is xp

    def test_initializer(self):
        # create objects using the Quantity constructor:
        q = Quantity(self.value, self.unit)
        assert q.value is self.value
        assert q.unit is self.unit
        q2 = Quantity(value=self.value, unit=self.unit)
        assert q2.value is self.value
        assert q2.unit is self.unit

    def test_need_value(self):
        with pytest.raises(TypeError):
            Quantity(unit=self.unit)

    def test_need_unit(self):
        with pytest.raises(TypeError):
            Quantity(self.value)

    def test_value_unit_immutable(self):
        q = Quantity(self.value, unit=self.unit)

        with pytest.raises(AttributeError):
            q.value = self.xp.asarray([10.0])

        with pytest.raises(AttributeError):
            q.unit = u.cm


class QuantityTestSetup:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a = cls.xp.asarray(np.arange(10.0).reshape(5, 2))
        cls.q = Quantity(cls.a, u.meter)


class QuantityAttributeTests(QuantityTestSetup):
    """Should follow Array API:
    https://data-apis.org/array-api/latest/API_specification/array_object.html#attributes
    """

    def test_deferred_to_value(self):
        value = self.q.value
        assert type(value) is self.ARRAY_CLASS
        assert self.q.shape == value.shape
        assert self.q.size == value.size
        assert self.q.ndim == value.ndim
        assert self.q.dtype == value.dtype
        assert self.q.device == array_api_compat.device(value)

    @pytest.mark.parametrize("transpose", ["mT", "T"])
    def test_transpose(self, transpose):
        try:
            getattr(self.a, transpose)
        except AttributeError:
            pytest.xfail(reason=f"{self.xp!r} does not have .{transpose}.")
        q_t = getattr(self.q, transpose)
        assert q_t.unit is self.q.unit
        assert type(q_t.value) is self.ARRAY_CLASS
        expected = getattr(self.q.value, transpose)
        assert_array_equal(q_t.value, expected)


class QuantityCopyTests(QuantityTestSetup):
    def test_copy(self):
        q_copy = copy.copy(self.q)
        assert q_copy is not self.q
        assert q_copy.value is self.q.value
        assert q_copy.unit is self.q.unit

    def test_deepcopy(self):
        try:
            copy.deepcopy(self.a)
        except TypeError:
            pytest.xfail(reason="cannot deepcopy {self.xp!r}.")
        q_dc = copy.deepcopy(self.q)
        assert q_dc is not self.q
        assert type(q_dc.value) is self.ARRAY_CLASS
        assert q_dc.value is not self.q.value
        assert q_dc.unit is self.q.unit  # u.m is always the same
        assert_array_equal(q_dc.value, self.q.value)


class QuantityMethodTests(QuantityTestSetup):
    """Test non-operator methods (for those, see test_operations).
    https://data-apis.org/array-api/latest/API_specification/array_object.html#methods
    This leaves:
    __array_namespace__
    __getitem__
    __setitem__
    to_device

    TODO: implemented and test the following
    __dlpack__
    __dlpack_device__
    """

    def test_array_namespace(self):
        assert self.q.__array_namespace__() is np

    def test_getitem(self):
        q2 = self.q[:2, :]  # Note Array API: need to specify both
        assert isinstance(q2, Quantity)
        assert type(q2.value) is self.ARRAY_CLASS
        assert q2.unit == u.meter
        assert q2.shape == (2, 2)
        assert_array_equal(q2.value, self.q.value[:2, :])

    def test_setitem(self):
        # Create explicitly to ensure we do not change self.q1.
        q = Quantity(self.xp.asarray(np.arange(10.0).reshape(5, 2)), self.q.unit)
        if self.NO_SETITEM:
            pytest.xfail(reason=f"array type {self.xp!r} elements cannot be set")
        q[:2, :] = Quantity(200.0, u.cm)
        assert q.unit is self.q.unit
        assert_array_equal(q.value[:2, :], 2.0)
        assert_array_equal(q.value[2:, :], self.q.value[2:, :])

    def test_to_device(self):
        q = self.q.to_device(self.q.device)
        assert q.unit is self.q.unit
        assert_array_equal(q.value, self.q.value)


# Create the actual test classes.
for base_setup in ARRAY_NAMESPACES:
    for tests in (
        QuantityCreationTests,
        QuantityAttributeTests,
        QuantityCopyTests,
        QuantityMethodTests,
    ):
        name = f"Test{tests.__name__}{base_setup.__name__}"
        globals()[name] = type(name, (tests, base_setup), {})
