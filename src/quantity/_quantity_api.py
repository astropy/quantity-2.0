"""Minimal definition of the Quantity API."""

__all__ = ["Quantity", "ArrayQuantity", "Unit"]

from typing import Protocol, runtime_checkable

from astropy.units import UnitBase as Unit

from ._array_api import Array


@runtime_checkable
class Quantity(Protocol):
    """Minimal definition of the Quantity API."""

    value: Array
    unit: Unit


@runtime_checkable
class ArrayQuantity(Quantity, Array, Protocol):
    """An array-valued Quantity."""

    ...
