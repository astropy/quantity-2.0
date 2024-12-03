"""Quantity-2.0: the Quantity API.

This module provides runtime-checkable Protocol objects that define the Quantity
API. In particular there are:

- `Quantity`: the minimal definition of a Quantity, separate from the Array API.
- `QuantityArray`: a Quantity that adheres to the Array API. This is the most
    complete definition of a Quantity, inheriting from the Quantity API and
    adding the requirements for the Array API.

"""
# Copyright (c) 2024 Astropy Developers. All rights reserved.

from ._src.api import Quantity, QuantityArray

__all__ = ["Quantity", "QuantityArray"]
