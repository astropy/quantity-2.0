# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class, creation and basic methods."""

from __future__ import annotations

import copy

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quantity import Quantity


class TestQuantityCreation:
    def test_initializer(self):
        # create objects using the Quantity constructor:
        value = np.arange(3.0)
        unit = u.m
        q = Quantity(value, unit)
        assert q.value is value
        assert q.unit is unit
        q2 = Quantity(value=value, unit=unit)
        assert q2.value is value
        assert q2.unit is u.m

    def test_need_value(self):
        with pytest.raises(TypeError):
            Quantity(unit=u.m)

    def test_need_unit(self):
        with pytest.raises(TypeError):
            Quantity(np.array([3.0]))

    def test_value_unit_immutable(self):
        q = Quantity(np.array([11.0]), unit=u.meter)

        with pytest.raises(AttributeError):
            q.value = np.array([10.0])

        with pytest.raises(AttributeError):
            q.unit = u.cm


class QuantityTestSetup:
    @classmethod
    def setup_class(cls):
        cls.q = Quantity(np.arange(10.0).reshape(5, 2), u.meter)


class TestQuantityAttributes(QuantityTestSetup):
    """Should follow Array API:
    https://data-apis.org/array-api/latest/API_specification/array_object.html#attributes
    """

    def test_deferred_to_value(self):
        value = self.q.value
        assert self.q.shape == value.shape
        assert self.q.size == value.size
        assert self.q.ndim == value.ndim
        assert self.q.dtype == value.dtype
        assert self.q.device == value.device

    @pytest.mark.parametrize("transpose", ["mT", "T"])
    def test_transpose(self, transpose):
        q_t = getattr(self.q, transpose)
        assert q_t.unit is self.q.unit
        expected = getattr(self.q.value, transpose)
        assert_array_equal(q_t.value, expected)


class TestCopy(QuantityTestSetup):
    def test_copy(self):
        q_copy = copy.copy(self.q)
        assert q_copy is not self.q
        assert q_copy.value is self.q.value
        assert q_copy.unit is self.q.unit

    def test_deepcopy(self):
        q_dc = copy.deepcopy(self.q)
        assert q_dc is not self.q
        assert q_dc.value is not self.q.value
        assert q_dc.unit is self.q.unit  # u.m is always the same
        assert_array_equal(q_dc.value, self.q.value)


class TestQuantityMethods(QuantityTestSetup):
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
        q2 = self.q[:2]
        assert isinstance(q2, Quantity)
        assert q2.unit == u.meter
        assert q2.shape == (2, 2)
        assert_array_equal(q2.value, self.q.value[:2])

    def test_setitem(self):
        q = copy.deepcopy(self.q)
        q[:2] = Quantity(200.0, u.cm)
        assert q.unit is self.q.unit
        assert_array_equal(q.value[:2], 2.0)
        assert_array_equal(q.value[2:], self.q.value[2:])

    def test_to_device(self):
        q = self.q.to_device("cpu")
        assert q.unit is self.q.unit
        assert_array_equal(q.value, self.q.value)
