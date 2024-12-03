"""Quantity-2.0: Prototyping the next generation Quantity.

Copyright (c) 2024 Astropy Developers. All rights reserved.
"""

from . import api
from ._src import Quantity
from .version import version as __version__  # noqa: F401

__all__ = [
    # modules
    "api",
    # functions and classes
    "Quantity",
]
