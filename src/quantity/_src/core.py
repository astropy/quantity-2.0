# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import array_api_compat
import astropy.units as u
import numpy as np
from astropy.units.quantity_helper import UFUNC_HELPERS, converters_and_unit

from .api import QuantityArray
from .utils import has_array_namespace

if TYPE_CHECKING:
    from typing import Any

    from .api import Unit
    from .array_api import Array


DIMENSIONLESS = u.dimensionless_unscaled

PYTHON_NUMBER = float | int | complex
TUPLE_OR_LIST = tuple | list

NORMALIZED_FUNCTION_NAMES = {
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "conjugate": "conj",
    "true_divide": "divide",
    "power": "pow",
}


def get_value_and_unit(
    arg: QuantityArray | Array, default_unit: Unit | None = None
) -> tuple[Array, Unit]:
    return (
        (arg.value, arg.unit) if isinstance(arg, QuantityArray) else (arg, default_unit)
    )


def value_in_unit(value, unit):
    v_value, v_unit = get_value_and_unit(value, default_unit=DIMENSIONLESS)
    return v_unit.to(unit, v_value)


_OP_TO_NP_FUNC = {
    "__add__": np.add,
    "__floordiv__": np.floor_divide,
    "__matmul__": np.matmul,
    "__mod__": np.mod,
    "__mul__": np.multiply,
    "__sub__": np.subtract,
    "__truediv__": np.true_divide,
}
OP_HELPERS = {op: UFUNC_HELPERS[np_func] for op, np_func in _OP_TO_NP_FUNC.items()}


def _make_op(fop, mode):
    assert mode in "fri"
    helper = OP_HELPERS[fop]
    op_func = getattr(operator, fop if mode != "i" else "__i" + fop[2:])
    if mode == "r":

        def wrapped_operator(u1, u2):
            return op_func(u2, u1)

        def wrapped_helper(u1, u2):
            convs, result_unit = helper(op_func, u2, u1)
            return convs[::-1], result_unit

    else:
        wrapped_operator = op_func

        def wrapped_helper(u1, u2):
            return helper(op_func, u1, u2)

    def __op__(self, other):
        return self._operate(other, wrapped_operator, wrapped_helper)

    return __op__


def _make_ops(op):
    return tuple(_make_op(op, mode) for mode in "fri")


def _make_comp(comp):
    def __comp__(self, other):
        try:
            other = value_in_unit(other, self.unit)
        except Exception:
            return NotImplemented
        return getattr(self.value, comp)(other)

    return __comp__


def _make_deferred(attr):
    # Use array_api_compat getter if available (size, device), since
    # some array formats provide inconsistent implementations.
    attr_getter = getattr(array_api_compat, attr, operator.attrgetter(attr))

    def deferred(self):
        return attr_getter(self.value)

    return property(deferred)


def _make_same_unit_method(attr):
    if array_api_func := getattr(array_api_compat, attr, None):

        def same_unit(self, *args, **kwargs):
            return replace(self, value=array_api_func(self.value, *args, **kwargs))

    else:

        def same_unit(self, *args, **kwargs):
            return replace(self, value=getattr(self.value, attr)(*args, **kwargs))

    return same_unit


def _make_same_unit_attribute(attr):
    attr_getter = getattr(array_api_compat, attr, operator.attrgetter(attr))

    def same_unit(self):
        return replace(self, value=attr_getter(self.value))

    return property(same_unit)


def _make_defer_dimensionless(attr):
    def defer_dimensionless(self):
        try:
            return getattr(self.unit.to(DIMENSIONLESS, self.value), attr)()
        except Exception as exc:
            raise TypeError from exc

    return defer_dimensionless


def _check_pow_args(exp, mod):
    if mod is not None:
        return NotImplemented

    if not isinstance(exp, PYTHON_NUMBER):
        try:
            exp = exp.__complex__()
        except Exception:
            try:
                return exp.__float__()
            except Exception:
                return NotImplemented

    return exp.real if exp.imag == 0 else exp


@dataclass(frozen=True, eq=False)
class Quantity:
    value: Any
    unit: u.UnitBase

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        # TODO: make our own?
        return np

    def _operate(self, other, op_func, units_helper):
        if not has_array_namespace(other) and not isinstance(other, PYTHON_NUMBER):
            # HACK: unit should take care of this!
            if not isinstance(other, u.UnitBase):
                return NotImplemented

            try:
                unit = op_func(self.unit, other)
            except Exception:
                return NotImplemented
            else:
                return replace(self, unit=unit)

        other_value, other_unit = get_value_and_unit(other)
        self_value = self.value
        (conv0, conv1), unit = units_helper(self.unit, other_unit)
        if conv0 is not None:
            self_value = conv0(self_value)
        if conv1 is not None:
            other_value = conv1(other_value)
        try:
            value = op_func(self_value, other_value)
        except TypeError:
            # Deal with the very unlikely case that other is an array type
            # that knows about Quantity, but cannot handle the array we carry.
            return NotImplemented
        return replace(self, value=value, unit=unit)

    # Operators (skipping ones that make no sense, like __and__);
    # __pow__ and __rpow__ need special treatment and are defined below.
    __add__, __radd__, __iadd__ = _make_ops("__add__")
    __floordiv__, __rfloordiv__, __ifloordiv__ = _make_ops("__floordiv__")
    __matmul__, __rmatmul__, __imatmul__ = _make_ops("__matmul__")
    __mod__, __rmod__, __imod__ = _make_ops("__mod__")
    __mul__, __rmul__, __imul__ = _make_ops("__mul__")
    __sub__, __rsub__, __isub__ = _make_ops("__sub__")
    __truediv__, __rtruediv__, __itruediv__ = _make_ops("__truediv__")

    # Comparisons
    __eq__ = _make_comp("__eq__")
    __ge__ = _make_comp("__ge__")
    __gt__ = _make_comp("__gt__")
    __le__ = _make_comp("__le__")
    __lt__ = _make_comp("__lt__")
    __ne__ = _make_comp("__ne__")

    # Attributes deferred to those of .value
    dtype = _make_deferred("dtype")
    device = _make_deferred("device")
    ndim = _make_deferred("ndim")
    shape = _make_deferred("shape")
    size = _make_deferred("size")

    # Deferred to .value, yielding new Quantity with same unit.
    mT = _make_same_unit_attribute("mT")
    T = _make_same_unit_attribute("T")
    __abs__ = _make_same_unit_method("__abs__")
    __neg__ = _make_same_unit_method("__neg__")
    __pos__ = _make_same_unit_method("__pos__")
    __getitem__ = _make_same_unit_method("__getitem__")
    to_device = _make_same_unit_method("to_device")

    # Deferred to .value, after making ourselves dimensionless (if possible).
    __complex__ = _make_defer_dimensionless("__complex__")
    __float__ = _make_defer_dimensionless("__float__")
    __int__ = _make_defer_dimensionless("__int__")

    # TODO: __dlpack__, __dlpack_device__

    def __pow__(self, exp, mod=None):
        exp = _check_pow_args(exp, mod)
        if exp is NotImplemented:
            return NotImplemented

        value = operator.__pow__(self.value, exp)
        return replace(self, value=value, unit=self.unit**exp)

    def __ipow__(self, exp, mod=None):
        exp = _check_pow_args(exp, mod)
        if exp is NotImplemented:
            return NotImplemented

        value = operator.__ipow__(self.value, exp)
        return replace(self, value=value, unit=self.unit**exp)

    def __setitem__(self, item, value):
        self.value[item] = value_in_unit(value, self.unit)

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        # Used for our namespace.
        if (where := kwargs.get("where")) is not None and isinstance(where, Quantity):
            return NotImplemented

        converters, unit = converters_and_unit(function, method, *inputs)
        out = kwargs.get("out")
        if out is not None:
            # If pre-allocated output is used, check it is suitable.
            # This also returns array view, to ensure we don't loop back.
            kwargs["out"] = self._check_output(
                out[0] if len(out) == 1 else out, unit, function=function
            )

        if method == "reduce" and "initial" in kwargs and unit is not None:
            # Special-case for initial argument for reductions like
            # np.add.reduce.  This should be converted to the output unit as
            # well, which is typically the same as the input unit (but can
            # in principle be different: unitless for np.equal, radian
            # for np.arctan2, though those are not necessarily useful!)
            kwargs["initial"] = self._to_own_unit(kwargs["initial"], unit=unit)

        input_values = [get_value_and_unit(in_)[0] for in_ in inputs]
        if not all(
            isinstance(v, PYTHON_NUMBER) or has_array_namespace(v) for v in input_values
        ):
            return NotImplemented
        input_values = [
            v if conv is None else conv(v)
            for (v, conv) in zip(input_values, converters, strict=True)
        ]
        try:
            xp = self.value.__array_namespace__()
        except AttributeError:
            try:
                xp = array_api_compat.array_namespace(self.value)
            except TypeError:
                xp = np

        if xp is np:
            xp_func = function
        else:
            function_name = NORMALIZED_FUNCTION_NAMES.get(n := function.__name__, n)
            xp_func = getattr(xp, function_name)
        try:
            result = getattr(xp_func, method)(*input_values, **kwargs)
        except Exception:
            # TODO: JAX supports "at" method if one passes in inplace=False.
            return NotImplemented
        return self._result_as_quantity(result, unit)

    @classmethod
    def _check_output(cls, output, unit, function=None):
        if isinstance(output, tuple):
            return tuple(
                cls._check_output(output_, unit_, function)
                for output_, unit_ in zip(output, unit, strict=True)
            )

        if output is None:
            return None

        if unit is None:
            if isinstance(output, Quantity):
                msg = "Cannot store non-quantity output{} in {} instance"
                raise TypeError(
                    msg.format(
                        (
                            f" from {function.__name__} function"
                            if function is not None
                            else ""
                        ),
                        type(output),
                    )
                )
            return output

        if not isinstance(output, Quantity):
            if unit == DIMENSIONLESS:
                return output

            msg = (
                "Cannot store output with unit '{}'{} in {} instance. "
                "Use {} instance instead."
            )
            raise u.UnitTypeError(
                msg.format(
                    unit,
                    (
                        f" from {function.__name__} function"
                        if function is not None
                        else ""
                    ),
                    type(output),
                    cls,
                )
            )
        return output.value

    @classmethod
    def _result_as_quantity(cls, result, unit):
        if isinstance(result, TUPLE_OR_LIST):
            # Some np.linalg functions return namedtuple, which is handy to access
            # elements by name, but cannot be directly initialized with an iterator.
            result_cls = getattr(result, "_make", result.__class__)
            return result_cls(cls(r, u) for (r, u) in zip(result, unit, strict=True))

        # If needed, weap the result array as a Quantity with the proper unit.
        return result if unit is None else cls(result, unit)

    __array_function__ = None
