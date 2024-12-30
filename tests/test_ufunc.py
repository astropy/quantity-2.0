# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test ufunc behaviour beyond what is required by the Array API."""

import astropy.units as u
import numpy as np
import pytest

from quantity import Quantity

from .conftest import ARRAY_NAMESPACES, assert_quantity_equal
from .test_element_wise_functions import QuantitySetup

# Ensure we test functions from our own array namespace
# (currently, just np, but may change).
qp = Quantity(np.array(1.0), u.one).__array_namespace__()


class Inplace(QuantitySetup):
    def try_func(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            if self.NO_OUTPUTS:
                pytest.xfail(reason="array type does not support out argument")

    @pytest.mark.parametrize("func", ["negative", "square"])
    def test_inplace_one_arg(self, func):
        qp_func = getattr(qp, func)
        exp = qp_func(self.q1)
        q_out = Quantity(self.xp.zeros_like(exp.value), exp.unit)
        got = self.try_func(qp_func, self.q1, out=q_out)
        assert got is not q_out  # Quantity is immutable.
        assert got.value is q_out.value
        assert_quantity_equal(got, exp)

    @pytest.mark.parametrize("func", ["add", "subtract", "divide"])
    def test_inplace_two_arg(self, func):
        qp_func = getattr(qp, func)
        exp = qp_func(self.q1, self.q2)
        q_out = Quantity(self.xp.zeros_like(exp.value), exp.unit)
        got = self.try_func(qp_func, self.q1, self.q2, out=q_out)
        assert got is not q_out  # Quantity is immutable.
        assert got.value is q_out.value
        assert_quantity_equal(got, exp)

    def test_inplace_two_outputs(self):
        if any(t in self.__class__.__name__ for t in ("APIStrict", "Dask")):
            pytest.xfail(reason=f"{self.q1.__class__} does not have divmod")
        exps = qp.divmod(self.q1, self.q2)
        q_outs = tuple(
            Quantity(self.xp.zeros_like(exp.value), exp.unit) for exp in exps
        )
        gots = self.try_func(qp.divmod, self.q1, self.q2, out=q_outs)
        for got, q_out, exp in zip(gots, q_outs, exps, strict=False):
            assert got is not q_out  # Quantity is immutable.
            assert got.value is q_out.value
            assert_quantity_equal(got, exp)


class Methods(QuantitySetup):
    def test_reduce(self):
        if not hasattr(self.xp.add, "reduce"):
            pytest.xfail("array type does not support ufunc.reduce")
        exp = Quantity(self.xp.add.reduce(self.a1, axis=0), self.q1.unit)
        got = qp.add.reduce(self.q1, axis=0)
        assert_quantity_equal(got, exp)

    def test_reduceat(self):
        if not hasattr(self.xp.add, "reduceat"):
            pytest.xfail("array type does not support ufunc.reduce_at")
        indices = np.array((2, 3))  # JAX only takes scalar or ndarray.
        exp = Quantity(self.xp.add.reduceat(self.a1, indices, axis=0), self.q1.unit)
        got = qp.add.reduceat(self.q1, indices, axis=0)
        assert_quantity_equal(got, exp)

    def test_accumulate(self):
        if not hasattr(self.xp.add, "accumulate"):
            pytest.xfail("array type does not support ufunc.accumulate")
        exp = Quantity(self.xp.add.accumulate(self.a1, axis=0), self.q1.unit)
        got = qp.add.accumulate(self.q1, axis=0)
        assert_quantity_equal(got, exp)

    def test_at(self):
        if not hasattr(self.xp.add, "at") or self.NO_OUTPUTS:
            # TODO: NO_OUTPUTS is not strictly applicable; e.g., JAX supports
            # np.add.at but one has to pass in inplace=False.
            pytest.xfail("array type does not support ufunc.at")
        values = [1.0, 2.0]
        a = self.xp.asarray(values)
        self.xp.add.at(a, 1, 100.0)
        exp = Quantity(a, u.cm)

        got = Quantity(self.xp.asarray(values), u.cm)
        qp.add.at(got, 1, Quantity(1.0, u.m))
        assert_quantity_equal(got, exp)


# Create the actual test classes.
for base_setup in ARRAY_NAMESPACES:
    for tests in (Inplace, Methods):
        name = f"Test{tests.__name__}{base_setup.__name__}"
        globals()[name] = type(name, (tests, base_setup), {})


def test_where_not_supported():
    q = Quantity(np.asarray([1.0, 2.0]), u.m)
    with pytest.raises(TypeError):
        np.add(q, q, where=q)
