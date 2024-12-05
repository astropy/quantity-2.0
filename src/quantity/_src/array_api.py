"""Minimal definition of the Array API.

NOTE: this module will be deprecated when
https://github.com/data-apis/array-api-typing is released.

"""

from __future__ import annotations

__all__ = ["HasArrayNameSpace", "Array"]

from typing import Any, Protocol, runtime_checkable


class HasArrayNameSpace(Protocol):
    """Minimal definition of the Array API."""

    def __array_namespace__(self) -> Any: ...


@runtime_checkable
class Array(HasArrayNameSpace, Protocol):
    """Minimal definition of the Array API."""

    def __pow__(self, other: Any) -> Array: ...
