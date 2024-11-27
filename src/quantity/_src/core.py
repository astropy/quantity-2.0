"""Quantity."""
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union, cast, overload

import array_api_compat
import astropy.units as u
import numpy as np
from astropy.units import UnitBase as Unit
from astropy.units.quantity_helper import UFUNC_HELPERS

from .api import QuantityArray
from .utils import has_array_namespace

if TYPE_CHECKING:
    from types import NotImplementedType
    from typing import Any, Self

    from ._array_api import Array
    from ._quantity_api import ArrayQuantity, Unit


T = TypeVar("T")

    from .api import Unit
    from .array_api import Array


DIMENSIONLESS = u.dimensionless_unscaled

PYTHON_NUMBER = float | int | complex


def get_value_and_unit(
    arg: QuantityArray | Array, default_unit: Unit | None = None
) -> tuple[Array, Unit]:
    return (
        (arg.value, arg.unit) if isinstance(arg, QuantityArray) else (arg, default_unit)
    )


def value_in_unit(value: Array, unit: Unit) -> Array:
    v_value, v_unit = get_value_and_unit(value, default_unit=DIMENSIONLESS)
    return cast(Array, v_unit.to(unit, v_value))


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


QuantityOpCallable: TypeAlias = Callable[
    ["Quantity", Any], Union["Quantity", NotImplementedType]
]


def _make_op(fop: str, mode: str) -> QuantityOpCallable:
    assert mode in "fri"
    op = fop if mode == "f" else "__" + mode + fop[2:]
    helper = OP_HELPERS[fop]
    op_func = getattr(operator, fop)
    if mode == "r":

        def wrapped_helper(u1, u2):
            return helper(op_func, u2, u1)
    else:

        def wrapped_helper(u1, u2):
            return helper(op_func, u1, u2)

    def __op__(self, other):
        return self._operate(other, op, wrapped_helper)

    return __op__


def _make_ops(
    op: str,
) -> tuple[QuantityOpCallable, QuantityOpCallable, QuantityOpCallable]:
    return (_make_op(op, "f"), _make_op(op, "r"), _make_op(op, "i"))


def _make_comp(comp: str) -> Callable[[Quantity, Any], Array]:
    def _comp_(self: Quantity, other: Any) -> Array | NotImplementedType:
        try:
            other = value_in_unit(other, self.unit)
        except Exception:
            return NotImplemented
        return getattr(self.value, comp)(other)

    return _comp_


def _make_deferred(attr: str) -> Callable[[Quantity], property]:
    # Use array_api_compat getter if available (size, device), since
    # some array formats provide inconsistent implementations.
    attr_getter = getattr(array_api_compat, attr, operator.attrgetter(attr))

    def deferred(self: Quantity):
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


# -----------------


@overload
def _parse_pow_mod(mod: None, /) -> None: ...


@overload
def _parse_pow_mod(mod: object, /) -> NotImplementedType: ...


def _parse_pow_mod(mod: T, /) -> T | NotImplementedType:
    return mod if mod is None else NotImplemented  # type: ignore[redundant-expr]


# -----------------


@overload
def _check_pow_exp(exp: Array | PYTHON_NUMBER, /) -> PYTHON_NUMBER: ...


@overload
def _check_pow_exp(exp: object, /) -> NotImplementedType: ...


def _check_pow_exp(exp: Any, /) -> PYTHON_NUMBER | NotImplementedType:
    out: PYTHON_NUMBER
    if isinstance(exp, PYTHON_NUMBER):
        out = exp
    else:
        try:
            out = complex(exp)
        except Exception:
            try:
                return float(exp)
            except Exception:
                return NotImplemented

    return out.real if out.imag == 0 else out


@dataclass(frozen=True, eq=False)
class Quantity:
    value: Array
    unit: Unit

    def __array_namespace__(self, *, api_version: str | None = None) -> Any:
        # TODO: make our own?
        del api_version
        return np

    def _operate(
        self, other: Any, op: Any, units_helper: Any
    ) -> Self | NotImplementedType:
        if not has_array_namespace(other) and not isinstance(other, PYTHON_NUMBER):
            # HACK: unit should take care of this!
            if not isinstance(other, u.UnitBase):
                return NotImplemented

            try:
                unit = getattr(operator, op)(self.unit, other)
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
            value = getattr(self_value, op)(other_value)
        except AttributeError:
            return NotImplemented
        if value is NotImplemented:
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

    def __pow__(self, exp: Any, mod: Any = None) -> Self | NotImplementedType:
        if (mod := _parse_pow_mod(mod)) is NotImplemented:
            return NotImplemented

        if (exp := _check_pow_exp(exp)) is NotImplemented:
            return NotImplemented

        value = self.value.__pow__(exp)
        if value is NotImplemented:
            return NotImplemented
        return replace(self, value=value, unit=self.unit**exp)

    def __ipow__(self, exp, mod=None):
        if (mod := _parse_pow_mod(mod)) is NotImplemented:
            return NotImplemented

        if (exp := _check_pow_exp(exp)) is NotImplemented:
            return NotImplemented

        value = self.value.__ipow__(exp)
        if value is NotImplemented:
            return NotImplemented
        return replace(self, value=value, unit=self.unit**exp)

    def __setitem__(self, item: Any, value: Any) -> None:
        """Call the setitem method of the array for the value in the unit.

        The Array API does not guarantee mutability of the underlying array,
        so this method will raise an exception if the array is immutable.

        """
        self.value[item] = value_in_unit(value, self.unit)  # type: ignore[index]

    __array_ufunc__ = None
    __array_function__ = None
