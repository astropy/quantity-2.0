# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class Array API compatibility."""

from __future__ import annotations

from quantity import Quantity

DEFER = {"dtype", "device", "ndim", "shape", "size"}
SAME_UNIT = {"mT", "T", "__abs__", "__neg__", "__pos__", "__getitem__", "to_device"}
implemented_ops = {"add", "floordiv", "matmul", "mod", "mul", "pow", "sub", "truediv"}
OPERATORS = {f"__{typ}{op}__" for op in implemented_ops for typ in ("", "r", "i")}
OPERATORS -= {"__rpow__"}
COMPARISONS = {"__eq__", "__ge__", "__gt__", "__le__", "__lt__", "__ne__"}
DEFER_DIMENSIONLESS = {"__complex__", "__float__", "__int__"}

meaningless_ops = {"and", "invert", "lshift", "or", "rshift", "xor"}
MEANINGLESS = {"__bool__", "__index__", "__rpow__"}
MEANINGLESS |= {f"__{typ}{op}__" for op in meaningless_ops for typ in ("", "r", "i")}
DLPACK = {"__dlpack__", "__dlpack_device__"}

REQUIRED = DEFER | SAME_UNIT | OPERATORS | COMPARISONS | DEFER_DIMENSIONLESS | DLPACK
NOT_YET_IMPLEMENTED = DLPACK


class TestQuantityArrayApiCompatibility:
    def test_method_compatibility(self):
        assert set(Quantity.__dict__) > (REQUIRED - NOT_YET_IMPLEMENTED)

    def test_no_meaningless_methods(self):
        assert not MEANINGLESS.intersection(Quantity.__dict__)
