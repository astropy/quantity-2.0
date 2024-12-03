"""Utility functions for the quantity package."""

import array_api_compat


def has_array_namespace(arg: object) -> bool:
    try:
        array_api_compat.array_namespace(arg)
    except TypeError:
        return False
    else:
        return True
