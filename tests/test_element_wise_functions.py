# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test that element-wise functions on Quantity properly propagate units.

This just tests the functions defined by the Array API:
https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html

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
    TrackingNameSpace,
    UsingNDArray,
    assert_quantity_equal,
)

# All element-wise functions defined by the Array API.
# https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html
ARRAY_API_ELEMENT_WISE_FUNCTIONS = {
    "abs", "acos", "acosh", "add", "asin", "asinh", "atan", "atan2", "atanh",
    "bitwise_and", "bitwise_left_shift", "bitwise_invert", "bitwise_or",
    "bitwise_right_shift", "bitwise_xor",
    "ceil", "clip", "conj", "copysign", "cos", "cosh",
    "divide", "equal", "exp", "expm1", "floor", "floor_divide",
    "greater", "greater_equal", "hypot", "imag", "isfinite", "isinf", "isnan",
    "less", "less_equal", "log", "log1p", "log2", "log10", "logaddexp",
    "logical_and", "logical_not", "logical_or", "logical_xor",
    "maximum", "minimum", "multiply", "negative", "not_equal", "positive",
    "pow", "real", "remainder", "round", "sign", "signbit", "sin", "sinh",
    "square", "sqrt", "subtract", "tan", "tanh", "trunc"
}  # fmt: skip


# Ensure we test functions from our own array namespace
# (currently, just np, but may change).
# Track which attributes where gotten so we can check our tests are complete.
qp = TrackingNameSpace(Quantity(np.array(1.0), u.one).__array_namespace__())


class QuantitySetup:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a1 = cls.xp.asarray(np.arange(1.0, 11.0).reshape(5, 2))
        cls.a2 = cls.xp.asarray([8.0, 10.0])
        cls.q1 = Quantity(cls.a1, u.meter)
        cls.q2 = Quantity(cls.a2, u.centimeter)


class Arithmetic(QuantitySetup):
    # Repeating QuantityOperationTests with corresponding functions.
    def test_add(self):
        # Take units from left object, q1
        got = qp.add(self.q1, self.q2)
        exp = Quantity(self.q1.value + self.q2.value / 100.0, u.m)
        assert_quantity_equal(got, exp, nulp=1)
        # Take units from left object, q2
        got = qp.add(self.q2, self.q1)
        exp = Quantity(self.q1.value * 100 + self.q2.value, u.cm)
        assert_quantity_equal(got, exp, nulp=1)

    def test_subtract(self):
        # Take units from left object, q1
        got = qp.subtract(self.q1, self.q2)
        exp = Quantity(self.q1.value - self.q2.value / 100.0, u.m)
        assert_quantity_equal(got, exp, nulp=1)

        # Take units from left object, q2
        got = qp.subtract(self.q2, self.q1)
        exp = Quantity(self.q2.value - 100.0 * self.q1.value, u.cm)
        assert_quantity_equal(got, exp, nulp=1)

    def test_multiply(self):
        got = qp.multiply(self.q1, self.q2)
        exp = Quantity(self.q1.value * self.q2.value, u.Unit("m cm"))
        assert_quantity_equal(got, exp)
        got = qp.multiply(self.q2, self.q1)
        assert_quantity_equal(got, exp)

    def test_divide(self):
        got = qp.divide(self.q1, self.q2)
        exp = Quantity(self.q1.value / self.q2.value, u.Unit("m/cm"))
        assert_quantity_equal(got, exp)
        got = qp.divide(self.q2, self.q1)
        exp = Quantity(self.q2.value / self.q1.value, u.Unit("cm/m"))
        assert_quantity_equal(got, exp)

    def test_floor_divide(self):
        got = qp.floor_divide(self.q1, self.q2)
        exp = Quantity(self.q1.value // (0.01 * self.q2.value), u.one)
        assert_quantity_equal(got, exp)
        got = qp.floor_divide(self.q2, self.q1)
        exp = Quantity(self.q2.value // (100.0 * self.q1.value), u.one)
        assert_quantity_equal(got, exp)

    def test_remainder(self):
        got = qp.remainder(self.q1, self.q2)
        exp = Quantity(self.q1.value % (0.01 * self.q2.value), self.q1.unit)
        assert_quantity_equal(got, exp)
        got = qp.remainder(self.q2, self.q1)
        exp = Quantity(self.q2.value % (100.0 * self.q1.value), self.q2.unit)
        assert_quantity_equal(got, exp)

    def test_negative(self):
        got = qp.negative(self.q1)
        exp = Quantity(-self.q1.value, u.m)
        assert_quantity_equal(got, exp)

        got = qp.negative(qp.negative(self.q1))
        assert_quantity_equal(got, self.q1)

    def test_positive(self):
        got = qp.positive(self.q1)
        assert_quantity_equal(got, self.q1)

    def test_abs(self):
        got = qp.abs(self.q1)
        exp = Quantity(abs(self.q1.value), u.m)
        assert_quantity_equal(got, exp)
        got = qp.abs(-self.q1)
        exp = Quantity(abs(self.q1.value), u.m)
        assert_quantity_equal(got, exp)

    def test_floor_divide_remainder_roundtrip(self):
        got = qp.add(
            qp.remainder(self.q1, self.q2),
            qp.multiply(qp.floor_divide(self.q1, self.q2), self.q2),
        )
        assert_quantity_equal(got, self.q1, nulp=1)
        got = qp.add(
            qp.remainder(self.q2, self.q1),
            qp.multiply(qp.floor_divide(self.q2, self.q1), self.q1),
        )
        assert_quantity_equal(got, self.q2, nulp=1)

    def test_dimensionless_operations(self):
        q1 = Quantity(self.a1, u.m / u.km)
        q2 = Quantity(self.a2, u.mm / u.km)
        got = qp.add(q1, q2)
        exp = Quantity(q1.value + q2.value / 1000.0, q1.unit)
        assert_quantity_equal(got, exp, nulp=1)
        # Test plain array.
        a = self.xp.asarray(1.0)
        got = qp.add(q1, a)
        exp = Quantity(q1.value / 1000.0 + 1.0, u.one)
        assert_quantity_equal(got, exp, nulp=1)

    def test_integer_promotion(self):
        a1 = self.xp.asarray([1, 2, 3])
        try:
            a1 * 0.001
        except Exception:
            pytest.xfail(reason="{self.xp!r} does not support int to float promotion.")
        q1 = Quantity(a1, u.m / u.km)
        a2 = self.xp.asarray([4, 5, 6])
        got = qp.add(q1, a2)
        exp = Quantity(q1.value / 1000.0 + a2, u.one)
        assert_quantity_equal(got, exp, nulp=1)

    def test_incompatible_units(self):
        """Raise when trying to add or subtract incompatible units"""
        q = Quantity(21.52, unit=u.second)
        with pytest.raises(u.UnitsError, match="[Cc]an only apply 'add' function"):
            qp.add(self.q1, q)

    def test_non_number_type(self):
        with pytest.raises(TypeError, match=r"[Uu]nsupported operand type\(s\).*"):
            qp.add(self.q1, {"a": 1})

        with pytest.raises(TypeError):
            qp.add(self.q1, u.meter)

    def test_multiplication_with_unit(self):
        with pytest.raises(TypeError):
            qp.multiply(self.q1, u.s)

        with pytest.raises(TypeError):
            qp.multiply(u.s, self.q1)

        with pytest.raises(TypeError):
            qp.multiply(self.q1, u.mag(u.Jy))

    def test_division_with_unit(self):
        with pytest.raises(TypeError):
            qp.divide(self.q1, u.s)

        with pytest.raises(TypeError):
            qp.divide(u.s, self.q1)

    def test_floor_division_errors(self):
        q2 = Quantity(self.a1, u.s)
        with pytest.raises(u.UnitsError, match="[Cc]an only apply 'floor_divide'"):
            qp.floor_divide(self.q1, q2)

        with pytest.raises(TypeError):
            qp.floor_divide(self.q1, u.s)

    def test_dimensionless_error(self):
        with pytest.raises(u.UnitsError):
            qp.add(self.q1, Quantity(self.a1, unit=u.one))

        with pytest.raises(u.UnitsError):
            qp.add(self.q1, Quantity(self.a1, unit=u.one))


class Powers(QuantitySetup):
    def test_pow(self):
        # raise quantity to a power
        p2 = self.xp.asarray(2.0)
        got = qp.pow(self.q1, p2)
        exp = Quantity(self.a1**2, u.Unit("m^2"))
        assert_quantity_equal(got, exp)
        p3 = self.xp.asarray(3.0)
        got = qp.pow(self.q1, p3)
        exp = Quantity(self.a1**3, u.Unit("m^3"))
        assert_quantity_equal(got, exp)

    def test_square(self):
        got = qp.square(self.q1)
        exp = Quantity(self.xp.square(self.a1), u.Unit("m^2"))
        assert_quantity_equal(got, exp)

    def test_sqrt(self):
        got = qp.sqrt(self.q1)
        exp = Quantity(self.xp.sqrt(self.a1), u.Unit("m^(1/2)"))
        assert_quantity_equal(got, exp)

    def test_hypot(self):
        got = qp.hypot(self.q1, self.q2)
        exp = Quantity(self.xp.hypot(self.a1, 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)


class ArithmeticWithNumbers(QuantitySetup):
    # Separate tests, since not strictly required by the Array API,
    # and hence array_api_strict doesn't pass with them.
    def test_multiplication_with_number(self):
        got = qp.multiply(15.0, self.q1)
        exp = Quantity(15.0 * self.q1.value, u.m)
        assert_quantity_equal(got, exp)
        got = qp.multiply(self.q1, 15.0)
        assert_quantity_equal(got, exp)

    def test_division_with_number(self):
        got = qp.divide(self.q1, 10.0)
        exp = Quantity(self.q1.value / 10.0, u.m)
        assert_quantity_equal(got, exp)
        got = qp.divide(11.0, self.q1)
        exp = Quantity(11.0 / self.q1.value, u.m**-1)
        assert_quantity_equal(got, exp)

    @pytest.mark.parametrize(
        "exponent",
        [2, 2.0, np.uint64(2), np.int32(2), np.float32(2), Quantity(2.0, u.one)],
    )
    def test_quantity_as_power(self, exponent):
        # raise unit to a dimensionless Quantity power
        if isinstance(exponent, Quantity):
            pytest.xfail(reason="cannot handle quantity exponent yet")
        got = qp.pow(self.q1, exponent)
        exp = Quantity(self.q1.value**2, u.m**2)
        assert_quantity_equal(got, exp)


class Comparisons(QuantitySetup):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.q1_in_cm = Quantity(cls.q1.value * 100.0, u.cm)
        cls.a2_in_m = cls.q2.unit.to(cls.q1.unit, cls.a2)

    @pytest.mark.parametrize(
        ("func", "op", "includes_equal"),
        [
            (qp.equal, operator.eq, True),
            (qp.not_equal, operator.ne, False),
            (qp.greater, operator.gt, False),
            (qp.greater_equal, operator.ge, True),
            (qp.less, operator.lt, False),
            (qp.less_equal, operator.le, True),
        ],
    )
    def test_comparison(self, func, op, includes_equal):
        got = func(self.q1, self.q1_in_cm)
        assert got.shape == self.q1.shape
        assert_array_equal(got, includes_equal)
        got = func(self.q1, self.q2)
        exp = op(self.q1.value, self.a2_in_m)
        assert_array_equal(got, exp)

    def test_not_equal_to_unit(self):
        unit = u.cm**3
        q = Quantity(self.xp.asarray([1.0]), unit)
        with pytest.raises(TypeError):
            qp.not_equal(q, unit)


class NumericTests:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a = cls.xp.asarray([1.1, 1.9, -2.1, np.inf, -np.inf, np.nan])
        cls.q = Quantity(cls.a, u.m)

    @pytest.mark.parametrize("func", ["isfinite", "isinf", "isnan", "sign", "signbit"])
    def test_numeric_test(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        got = qp_func(self.q)
        exp = xp_func(self.a)
        assert not isinstance(got, Quantity)
        assert_array_equal(got, exp)


class ClipAndTransfer:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a1 = cls.xp.asarray([1.1, 1.9, -2.1, np.inf, -np.inf, np.nan])
        cls.q1 = Quantity(cls.a1, u.m)
        cls.a2 = cls.xp.asarray([1.2, 180.0, -200.0, -np.inf, np.nan, np.nan])
        cls.q2 = Quantity(cls.a2, u.cm)

    @pytest.mark.parametrize("func", ["ceil", "floor", "round", "trunc"])
    def test_one_arg(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        if not isinstance(qp_func, np.ufunc):
            pytest.xfail(reason="only numpy ufuncs are supported")
        got = qp_func(self.q1)
        exp = Quantity(xp_func(self.a1), self.q1.unit)
        assert_quantity_equal(got, exp)

    @pytest.mark.parametrize("func", ["minimum", "maximum"])
    def test_min_max(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        got = qp_func(self.q1, self.q2)
        exp = Quantity(xp_func(self.a1, 0.01 * self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)

    def test_copysign(self):
        got = qp.copysign(self.q1, self.q2)
        exp = Quantity(self.xp.copysign(self.a1, self.a2), self.q1.unit)
        assert_quantity_equal(got, exp)

    @pytest.mark.xfail(reason="only numpy ufuncs are supported")
    def test_clip(self):
        q3 = Quantity(self.xp.asarray(1.0), u.km)
        got = qp.clip(self.q1, min=self.q2, max=q3)
        exp = Quantity(
            self.xp.clip(self.a1, min=0.01 * self.a2, max=1000.0), self.q1.unit
        )
        assert_quantity_equal(got, exp)


class Trig:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        angles = [-45.0, 0.0, 30.0, 120.0]
        cls.a_deg = cls.xp.asarray(angles)
        cls.a_rad = cls.xp.asarray(np.deg2rad(angles))
        cls.q_deg = Quantity(cls.a_deg, u.deg)
        cls.q_rad = Quantity(cls.a_rad, u.rad)

    @pytest.mark.parametrize("func", ["sin", "cos", "tan", "sinh", "cosh", "tanh"])
    def test_trig(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        got = qp_func(self.q_deg)
        exp = Quantity(xp_func(self.a_rad), u.one)
        assert_quantity_equal(got, exp, nulp=1)

    @pytest.mark.parametrize(
        "func", ["asin", "acos", "atan", "asinh", "acosh", "atanh"]
    )
    def test_inverse(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)

        xp_forw = getattr(self.xp, func[1:])
        a_in = xp_forw(self.a_rad)
        q_in = Quantity(a_in * 100.0, u.percent)

        got = qp_func(q_in)
        exp = Quantity(xp_func(a_in), u.rad)
        assert_quantity_equal(got, exp, nulp=5)

    def test_atan2(self):
        sina = qp.sin(self.q_deg)
        cosa = qp.cos(self.q_deg)
        got = qp.atan2(sina, cosa)
        exp = Quantity(
            self.xp.atan2(self.xp.sin(self.a_rad), self.xp.cos(self.a_rad)), u.rad
        )
        assert_quantity_equal(got, exp, nulp=1)


class ExpAndLog:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a = cls.xp.asarray([0.5, 1.0, 2.0])
        cls.q = Quantity(cls.a * 100.0, u.percent)

    @pytest.mark.parametrize("func", ["exp", "expm1", "log", "log1p", "log2", "log10"])
    def test_exp_or_log(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        got = qp_func(self.q)
        exp = Quantity(xp_func(self.a), u.one)
        assert_quantity_equal(got, exp, nulp=1)

    def test_logaddexp(self):
        q2 = Quantity(self.a, u.one)
        got = qp.logaddexp(self.q, q2)
        exp = Quantity(self.xp.logaddexp(self.a, self.a), u.one)
        assert_quantity_equal(got, exp, nulp=1)


class Complex:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.a = cls.xp.asarray([1 + 1j, 1.0, -2.0 + 1.0j, np.inf, -np.inf * 1j])
        cls.q = Quantity(cls.a, u.m)

    @pytest.mark.parametrize("func", ["conj", "imag", "real"])
    def test_func(self, func):
        qp_func = getattr(qp, func)
        xp_func = getattr(self.xp, func)
        if not isinstance(qp_func, np.ufunc):
            pytest.xfail(reason="only numpy ufuncs are supported")
        got = qp_func(self.q)
        exp = Quantity(xp_func(self.a), self.q.unit)
        assert_quantity_equal(got, exp)


# Create the actual test classes.
for base_setup in ARRAY_NAMESPACES:
    for tests in (
        Arithmetic,
        Powers,
        ArithmeticWithNumbers,
        Comparisons,
        NumericTests,
        ClipAndTransfer,
        Trig,
        ExpAndLog,
        Complex,
    ):
        if tests is ArithmeticWithNumbers and base_setup.xp is array_api_strict:
            continue
        name = f"Test{tests.__name__}{base_setup.__name__}"
        globals()[name] = type(name, (tests, base_setup), {})


class TestUnsupported(QuantitySetup, UsingNDArray):
    """Unsupported functions. No need to test with anything but numpy."""

    @pytest.mark.parametrize(
        "func",
        [
            "bitwise_and", "bitwise_invert", "bitwise_or", "bitwise_xor",
            "bitwise_left_shift", "bitwise_right_shift",
            "logical_and", "logical_not", "logical_or", "logical_xor",
        ]
    )  # fmt: skip
    def test_unsupported(self, func):
        qp_func = getattr(qp, func)
        with pytest.raises(TypeError):
            qp_func(self.q1, self.q2)


def test_completeness():
    assert qp.used_attrs == ARRAY_API_ELEMENT_WISE_FUNCTIONS
