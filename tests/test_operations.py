# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test that operations on Quantity properly propagate units.

Note: tests classes are combined with setups for different array types
at the very end.  Hence, they do not have the usual Test prefix.
"""

import operator

import array_api_strict
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quantity import Quantity

from .conftest import (
    ARRAY_NAMESPACES,
    UsingArrayAPIStrict,
    UsingJAX,
    assert_quantity_equal,
)


class QuantitySetup:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a1 = cls.xp.asarray(np.arange(1.0, 11.0).reshape(5, 2))
        cls.a2 = cls.xp.asarray([8.0, 10.0])
        cls.q1 = Quantity(cls.a1, u.meter)
        cls.q2 = Quantity(cls.a2, u.centimeter)


class QuantityOperationTests(QuantitySetup):
    def test_addition(self):
        # Take units from left object, q1
        got = self.q1 + self.q2
        exp = Quantity(self.q1.value + self.q2.value / 100.0, u.m)
        assert_quantity_equal(got, exp, nulp=1)
        # Take units from left object, q2
        got = self.q2 + self.q1
        exp = Quantity(self.q1.value * 100 + self.q2.value, u.cm)
        assert_quantity_equal(got, exp, nulp=1)

    def test_subtraction(self):
        # Take units from left object, q1
        got = self.q1 - self.q2
        exp = Quantity(self.q1.value - self.q2.value / 100.0, u.m)
        assert_quantity_equal(got, exp, nulp=1)

        # Take units from left object, q2
        got = self.q2 - self.q1
        exp = Quantity(self.q2.value - 100.0 * self.q1.value, u.cm)
        assert_quantity_equal(got, exp, nulp=1)

    def test_multiplication(self):
        got = self.q1 * self.q2
        exp = Quantity(self.q1.value * self.q2.value, u.Unit("m cm"))
        assert_quantity_equal(got, exp)
        got = self.q2 * self.q1
        assert_quantity_equal(got, exp)

    def test_multiplication_with_number(self):
        got = 15.0 * self.q1
        exp = Quantity(15.0 * self.q1.value, u.m)
        assert_quantity_equal(got, exp)
        got = self.q1 * 15.0
        assert_quantity_equal(got, exp)

    def test_multiplication_with_unit(self):
        got = self.q1 * u.s
        exp = Quantity(self.q1.value, u.Unit("m s"))
        assert_quantity_equal(got, exp)

    def test_reverse_multiplication_with_unit(self):
        got = u.s * self.q1
        if isinstance(got, u.Quantity):
            pytest.xfail(reason="Astropy unit took over, causing u.Quantity return.")
        # TODO: for array-api-strict, this is not great, since it changes it
        # to a regular array. But the problem really is with astropy unit.
        exp = Quantity(np.float64(1.0) * self.q1.value, u.Unit("m s"))
        assert_quantity_equal(got, exp)

    def test_division(self):
        got = self.q1 / self.q2
        exp = Quantity(self.q1.value / self.q2.value, u.Unit("m/cm"))
        assert_quantity_equal(got, exp)
        got = self.q2 / self.q1
        exp = Quantity(self.q2.value / self.q1.value, u.Unit("cm/m"))
        assert_quantity_equal(got, exp)

    def test_division_with_number(self):
        got = self.q1 / 10.0
        exp = Quantity(self.q1.value / 10.0, u.m)
        assert_quantity_equal(got, exp)
        got = 11.0 / self.q1
        exp = Quantity(11.0 / self.q1.value, u.m**-1)
        assert_quantity_equal(got, exp)

    def test_division_with_unit(self):
        got = self.q1 / u.s
        exp = Quantity(self.q1.value, u.Unit("m/s"))
        assert_quantity_equal(got, exp)

    def test_reverse_division_with_unit(self):
        # Divide into a unit.
        got = u.s / self.q1
        if isinstance(got, u.Quantity):
            pytest.xfail(reason="Astropy unit took over, causing u.Quantity return.")
        # TODO: for array-api-strict, this is not great, since it changes it
        # to a regular array. But the problem really is with astropy unit.
        exp = Quantity(np.float64(1.0) / self.q1.value, u.Unit("s/m"))
        assert_quantity_equal(got, exp)

    def test_floor_division(self):
        got = self.q1 // self.q2
        exp = Quantity(self.q1.value // (0.01 * self.q2.value), u.one)
        assert_quantity_equal(got, exp)
        got = self.q2 // self.q1
        exp = Quantity(self.q2.value // (100.0 * self.q1.value), u.one)
        assert_quantity_equal(got, exp)

    def test_floor_division_errors(self):
        q2 = Quantity(self.a1, u.s)
        with pytest.raises(u.UnitsError, match="[Cc]an only apply 'floordiv'"):
            self.q1 // q2
        with pytest.raises(TypeError):
            self.q1 // u.s

    def test_mod(self):
        got = self.q1 % self.q2
        exp = Quantity(self.q1.value % (0.01 * self.q2.value), self.q1.unit)
        assert_quantity_equal(got, exp)
        got = self.q2 % self.q1
        exp = Quantity(self.q2.value % (100.0 * self.q1.value), self.q2.unit)
        assert_quantity_equal(got, exp)

    def test_floor_div_mod_roundtrip(self):
        got = self.q1 % self.q2 + (self.q1 // self.q2) * self.q2
        assert_quantity_equal(got, self.q1, nulp=1)
        got = self.q2 % self.q1 + (self.q2 // self.q1) * self.q1
        assert_quantity_equal(got, self.q2, nulp=1)

    def test_power(self):
        # raise quantity to a power
        got = self.q1**2
        exp = Quantity(self.q1.value**2, u.Unit("m^2"))
        assert_quantity_equal(got, exp)
        got = self.q1**3
        exp = Quantity(self.q1.value**3, u.Unit("m^3"))
        assert_quantity_equal(got, exp)

    @pytest.mark.parametrize(
        "exponent",
        [2, 2.0, np.uint64(2), np.int32(2), np.float32(2), Quantity(2.0, u.one)],
    )
    def test_quantity_as_power(self, exponent):
        # raise unit to a dimensionless Quantity power
        got = self.q1**exponent
        exp = Quantity(self.q1.value**2, u.m**2)
        assert_quantity_equal(got, exp)

    def test_matmul(self):
        a = self.xp.eye(3)
        q = Quantity(a, u.m)
        got = q @ a
        exp = Quantity(a, u.m)
        assert_quantity_equal(got, exp)
        got = a @ q
        assert_quantity_equal(got, exp)
        got = q @ q
        exp = Quantity(a, u.m**2)
        assert_quantity_equal(got, exp)
        a2 = self.xp.asarray(
            [[[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]],
             [[0., 1., 0.],
              [0., 0., 1.],
              [1., 0., 0.]],
             [[0., 0., 1.],
              [1., 0., 0.],
              [0., 1., 0.]]]
        )  # fmt: skip
        q2 = Quantity(a2, u.s**-1)
        got = q @ q2
        exp = Quantity(q.value @ q2.value, u.Unit("m/s"))
        assert_quantity_equal(got, exp)

    def test_negative(self):
        got = -self.q1
        exp = Quantity(-self.q1.value, u.m)
        assert_quantity_equal(got, exp)

        got = -(-self.q1)  # noqa: B002
        assert_quantity_equal(got, self.q1)

    def test_positive(self):
        got = +self.q1
        assert_quantity_equal(got, self.q1)

    def test_abs(self):
        got = abs(self.q1)
        exp = Quantity(abs(self.q1.value), u.m)
        assert_quantity_equal(got, exp)
        got = abs(-self.q1)
        exp = Quantity(abs(self.q1.value), u.m)
        assert_quantity_equal(got, exp)

    def test_incompatible_units(self):
        """Raise when trying to add or subtract incompatible units"""
        q = Quantity(21.52, unit=u.second)
        with pytest.raises(u.UnitsError, match="[Cc]an only apply 'add' function"):
            self.q1 + q

    def test_non_number_type(self):
        with pytest.raises(TypeError, match=r"[Uu]nsupported operand type\(s\).*"):
            self.q1 + {"a": 1}

        with pytest.raises(TypeError):
            self.q1 + u.meter

        with pytest.raises(TypeError):
            self.q1 * u.mag(u.Jy)

        with pytest.raises(TypeError):
            self.q1**u.m

    def test_pow_mod_not_supported(self):
        with pytest.raises(TypeError):
            pow(self.q1, 2, 1.0)

    def test_dimensionless_operations(self):
        q1 = Quantity(self.a1, u.m / u.km)
        q2 = Quantity(self.a2, u.mm / u.km)
        got = q1 + q2
        exp = Quantity(q1.value + q2.value / 1000.0, q1.unit)
        assert_quantity_equal(got, exp, nulp=1)
        # Test plain float.
        got = q1 + 1.0
        exp = Quantity(q1.value / 1000.0 + 1.0, u.one)
        assert_quantity_equal(got, exp, nulp=1)

    def test_dimensionless_error(self):
        with pytest.raises(u.UnitsError):
            self.q1 + Quantity(self.a1, unit=u.one)

        with pytest.raises(u.UnitsError):
            self.q1 - Quantity(self.a1, unit=u.one)

    def test_integer_promotion(self):
        a1 = self.xp.asarray([1, 2, 3])
        try:
            a1 * 0.001
        except Exception:
            pytest.xfail(reason="{self.xp!r} does not support int to float promotion.")
        q1 = Quantity(a1, u.m / u.km)
        a2 = self.xp.asarray([4, 5, 6])
        got = q1 + a2
        exp = Quantity(q1.value / 1000.0 + a2, u.one)
        assert_quantity_equal(got, exp, nulp=1)

    def test_eq_ne(self):
        # equality/ non-equality is straightforward for quantity objects
        q = Quantity(self.q1.value * 100.0, u.cm)
        got = self.q1 == q
        assert got.shape == self.q1.shape
        assert_array_equal(got, True)
        got = self.q1 != q
        assert_array_equal(got, False)
        q = Quantity(self.q1.value * 10.0, u.cm)
        got = self.q1 == q
        assert_array_equal(got, False)
        got = self.q1 != q
        assert_array_equal(got, True)

    def test_not_equal_to_unit(self):
        # This should not work (unlike for astropy Quantity)
        unit = u.cm**3
        q = Quantity(self.xp.asarray([1.0]), unit)
        assert q != unit

    @pytest.mark.parametrize(
        ("value", "unit"),
        [(1.0, u.cm), (1.0, u.one), (0.0, u.cm), (0.0, u.one)],
    )
    def test_always_truthy(self, value, unit):
        q = Quantity(self.xp.asarray(value), unit)
        assert bool(q)  # default python behaviour when __bool__ is not present.

    @pytest.mark.parametrize(("value", "unit"), [(1.23, u.one), (1.1, u.m / u.km)])
    def test_numeric_converters(self, value, unit):
        # float and int should only work for scalar dimensionless quantities.
        q = Quantity(self.xp.asarray(value), unit)
        assert float(q) == float(q.unit.to(u.one, q.value))
        assert int(q) == int(q.unit.to(u.one, q.value))

        with pytest.raises(TypeError):
            operator.index(q)

    def test_numeric_converters_fail_on_non_dimenionless(self):
        q = Quantity(self.xp.asarray(1.0), u.m)
        with pytest.raises(TypeError):
            float(q)
        with pytest.raises(TypeError):
            int(q)

    def test_numeric_converters_fail_on_non_scalar(self):
        q = Quantity(self.xp.asarray([1.0, 2.0]), u.m)
        with pytest.raises(TypeError):
            float(q)
        with pytest.raises(TypeError):
            int(q)

    @pytest.mark.parametrize("value", [[1.0], np.arange(10.0)])
    def test_inplace(self, value):
        value = self.xp.asarray(value)
        s = Quantity(self.xp.asarray(value, copy=True), u.cycle)
        check = s
        s /= 2.0
        assert check.value is s.value or self.IMMUTABLE
        exp = Quantity(value / 2.0, u.cycle)
        assert_quantity_equal(s, exp)
        check = s
        s /= u.s
        assert check.value is s.value
        # Choice for making Quantity itself immutable.
        assert check.unit == u.cycle
        assert s.unit == u.cycle / u.s
        check = s
        s *= Quantity(self.xp.asarray(2.0), u.s)
        assert check.value is s.value or self.IMMUTABLE
        exp = Quantity(value, u.cycle)
        assert_quantity_equal(s, exp)

    def test_inplace_pow(self):
        value = self.xp.asarray([2.0])
        s = Quantity(self.xp.asarray(value, copy=True), u.cycle)
        check = s
        s **= 2.0
        assert check.value is s.value or self.IMMUTABLE
        exp = Quantity(value**2.0, u.cycle**2)
        assert_quantity_equal(s, exp)

        with pytest.raises(TypeError):
            s **= u.m


# Create the actual test classes.
for base_setup in ARRAY_NAMESPACES:
    for tests in (QuantityOperationTests,):
        name = f"Test{tests.__name__}{base_setup.__name__}"
        globals()[name] = type(name, (tests, base_setup), {})


class TestQuantitySubclassAndMixedArrayTypes(UsingArrayAPIStrict):
    """Subclass should take precedence, so execute reverse operations.

    Also use fact that array_api_strict does not recognize numpy arrays,
    while the other way around works.
    """

    def setup_class(cls):
        super().setup_class()

        class MyQuantity(Quantity):
            def __rsub__(self, other):  # Override to get precedence.
                return super().__rsub__(other)

        cls.MyQuantity = MyQuantity
        cls.a1 = cls.xp.asarray(np.arange(1.0, 11.0).reshape(5, 2))
        cls.a2 = cls.xp.asarray([8.0, 10.0])
        cls.q1 = Quantity(cls.a1, u.one)
        cls.q2 = Quantity(cls.a2, u.percent)
        cls.mq2 = MyQuantity(cls.a2, u.percent)

    def test_array_mix1(self):
        # Use ndarray for second regular Quantity
        a2 = np.asarray(self.a2)
        q2 = Quantity(a2, u.percent)
        got = self.q1 - q2
        exp = Quantity(np.asarray(self.a1 - 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)
        got = self.a1 - q2
        assert_quantity_equal(got, exp)

    def test_array_mix2(self):
        # Use ndarray for first Quantity and make second regular Quantity.
        a1 = np.asarray(self.a1)
        q1 = Quantity(a1, u.one)
        q2 = Quantity(self.a2, u.percent)
        got = q1 - q2
        exp = Quantity(np.asarray(self.a1 - 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)

    @pytest.mark.xfail(
        reason="strict_array_api.subtract behaves differently from __sub__"
    )
    def test_array_mix3(self):
        a1 = np.asarray(self.a1)
        got = a1 - self.q2
        exp = self.MyQuantity(np.asarray(self.a1 - 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)

    def test_subtract(self):
        got = self.q1 - self.mq2
        assert isinstance(got, self.MyQuantity)
        exp = self.MyQuantity(self.a1 - 0.01 * self.a2, self.q1.unit)
        assert_quantity_equal(got, exp)

    def test_subtract_mix(self):
        # Give q2 precedence, but ensure its contents returns NotImplemented;
        # this partially to cover all paths in Quantity._operate.
        a1 = np.asarray(self.a1)
        q1 = Quantity(a1, u.one)
        assert self.a2.__rsub__(a1) is NotImplemented
        got = q1 - self.mq2
        assert isinstance(got, Quantity)
        exp = Quantity(np.asarray(self.a1 - 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)


class TestIncompatibleArrays(UsingJAX):
    """Test propagation and cover an unlikely branch in Quantity._operate."""

    def test_incompatible_arrays(self):
        a1 = self.xp.asarray(1.0)
        a2 = array_api_strict.asarray(1.0)
        with pytest.raises(TypeError):  # Sanity check
            a1 + a2
        q1 = Quantity(a1, u.m)
        q2 = Quantity(a2, u.cm)
        with pytest.raises(TypeError):
            q1 + q2
