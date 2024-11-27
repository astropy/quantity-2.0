"""Minimal definition of the Array API."""

from __future__ import annotations

from typing import Any, Protocol


class HasArrayNameSpace(Protocol):
    """Minimal defintion of the Array API."""

    def __array_namespace__(self) -> Any: ...


class Array(HasArrayNameSpace, Protocol):
    """Minimal defintion of the Array API."""

    def __pow__(self, other: Any) -> Array: ...
