# Licensed under a 3-clause BSD style license - see LICENSE.rst

import array_api_compat
import astropy.units as u
import numpy as np
from astropy.utils.decorators import classproperty
from numpy.testing import assert_array_almost_equal_nulp, assert_array_equal

ARRAY_NAMESPACES = []


class ANSTests:
    IMMUTABLE = False  # default
    NO_SETITEM = False
    NO_OUTPUTS = False

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
    NO_OUTPUTS = True

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
    NO_OUTPUTS = True

    @classproperty(lazy=True)
    def xp(cls):
        return __import__("jax").numpy


def assert_quantity_equal(q1, q2, nulp=0):
    assert q1.unit == q2.unit
    assert q1.value.__class__ is q2.value.__class__
    if nulp:
        assert_array_almost_equal_nulp(q1.value, q2.value, nulp=nulp)
    else:
        assert_array_equal(q1.value, q2.value)


class TrackingNameSpace:
    """Intermediate namespace that tracks attributes that were used.

    Used to check whether we test complete sets of functions in the Array API.
    """

    def __init__(self, ns):
        self.ns = ns
        self.used_attrs = set()

    def __getattr__(self, attr):
        if not attr.startswith("_"):
            self.used_attrs.add(attr)
        return getattr(self.ns, attr)
