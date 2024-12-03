"""The Quantity API. Private module."""

__all__ = ["Quantity", "QuantityArray", "Unit"]

from typing import Protocol, runtime_checkable

from astropy.units import UnitBase as Unit

from .array_api import Array


@runtime_checkable
class Quantity(Protocol):
    """Minimal definition of the Quantity API.

    At minimum a Quantity must have the following attributes:

    - `value`: the numerical value of the quantity (adhering to the Array API)
    - `unit`: the unit of the quantity

    In practice, Quantities themselves must adhere to the Array API, not just
    their values. This stricter requirement is described by the `QuantityArray`
    protocol.

    See Also
    --------
    QuantityArray : A Quantity that adheres to the Array API

    """

    #: The numerical value of the quantity, adhering to the Array API.
    value: Array

    #: The unit of the quantity.
    unit: Unit


@runtime_checkable
class QuantityArray(Quantity, Array, Protocol):
    """An array-valued Quantity.

    A QuantityArray is a Quantity that itself adheres to the Array API. This
    means that the QuantityArray has properties like `shape`, `dtype`, and the
    `__array_namespace__` method, among many other properties and methods. To
    understand the full requirements of the Array API, see the `Array` protocol.
    The `Quantity` protocol describes the minimal requirements for a Quantity,
    separate from the Array API. QuantityArray is the combination of these two
    protocols and is the most complete description of a Quantity.

    See Also
    --------
    Quantity : The minimal Quantity API, separate from the Array API

    """

    ...
