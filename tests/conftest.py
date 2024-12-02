# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import annotations

import array_api_compat
import astropy.units as u
import numpy as np
from astropy.utils.decorators import classproperty

ARRAY_NAMESPACES = []


class ANSTests:
    IMMUTABLE = False  # default
    NO_SETITEM = False

    def __init_subclass__(cls, **kwargs):
        # Add class to namespaces available for testing if the underlying
        # array class is available.
        if not cls.__name__.startswith("Test"):
            try:
                cls.xp  # noqa: B018
            except ImportError:
                pass
            else:
                ARRAY_NAMESPACES.append(cls)

    @classmethod
    def setup_class(cls):
        cls.ARRAY_CLASS = type(cls.xp.ones((1,)))


class UsingNDArray(ANSTests):
    xp = np


class MonkeyPatchUnitConversion:
    @classmethod
    def setup_class(cls):
        super().setup_class()
        # TODO: update astropy so this monkeypatch is not necessary!
        # Enable non-coercing unit conversion on all astropy versions.
        cls._old_condition_arg = u.core._condition_arg
        u.core._condition_arg = lambda x: x

    @classmethod
    def teardown_class(cls):
        u.core._condition_arg = cls._old_condition_arg


class UsingArrayAPIStrict(MonkeyPatchUnitConversion, ANSTests):
    @classproperty(lazy=True)
    def xp(cls):
        return __import__("array_api_strict")


class UsingDask(MonkeyPatchUnitConversion, ANSTests):
    IMMUTABLE = True

    @classproperty(lazy=True)
    def xp(cls):
        import dask.array as da

        return array_api_compat.array_namespace(da.array([1.0]))


class UsingJAX(MonkeyPatchUnitConversion, ANSTests):
    IMMUTABLE = True
    NO_SETITEM = True

    @classproperty(lazy=True)
    def xp(cls):
        return __import__("jax").numpy
