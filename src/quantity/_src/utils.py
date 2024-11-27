"""Utility functions for the quantity package."""

from typing import Any, TypeGuard

import array_api_compat


def has_array_namespace(arg: Any) -> TypeGuard[Array]:
    try:
        array_api_compat.array_namespace(arg)
    except TypeError:
        return False
    return True
